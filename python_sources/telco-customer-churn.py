#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Companies providing services such as telecommunication, internet, TV  or insurances often have to deal with customer attrition, also known as customer churn. Customer churning could provide a significant loss in income for the companies as firstly, the company would no longer be able to generate income from the customer and the cost of attracting the customer in the first place would be wasted. Secondly, a loss in market share would not reflect well for the companies. 

# # Table of content

# - <a href='#1'>1. Import and view dataset</a>
#     - <a href='#1.1'>1.1. Data overview</a>
# - <a href='#2'>2. Exploratory analysis</a>
#     - <a href='#2.1'>2.1 Target variable - Churn</a>
#     - <a href='#2.2'>2.2 Gender</a>
#     - <a href='#2.3'>2.3 Senior citizen</a>
#     - <a href='#2.4'>2.4 Partner</a>
#     - <a href='#2.5'>2.5 Dependents</a>
#     - <a href='#3.6'>2.6 Tenure</a>
#     - <a href='#2.7'>2.7 PhoneService</a>
#     - <a href='#2.8'>2.8 MultipleLines</a>
#     - <a href='#2.9'>2.9 InternetService</a>
#     - <a href='#2.10'>2.10 OnlineSecurity</a>
#     - <a href='#2.11'>2.11 OnlineBackup</a>
#     - <a href='#2.12'>2.12 DeviceProtection</a>
#     - <a href='#2.13'>2.13 TechSupport</a>
#     - <a href='#2.14'>2.14 StreamingTV</a>
#     - <a href='#2.15'>2.15 StreamingMovies</a>
#     - <a href='#2.16'>2.16 Contract</a>
#     - <a href='#2.17'>2.17 PaperlessBilling</a>
#     - <a href='#2.18'>2.18 PaymentMethod</a>
#     - <a href='#2.19'>2.19 MonthlyCharges</a>
#     - <a href='#2.20'>2.20 TotalCharges</a>
# - <a href='#3'>3. Feature engineering</a>
#     - <a href='#3.1'>3.1 Ratio of past monthly charges to current month charge</a>
#     - <a href='#3.2'>3.2 Count of extra services</a>
# - <a href='#4'>4. Split into train and test set</a>
# - <a href='#5'>5. Data Preprocessing</a>
# - <a href='#6'>6. Feature importance</a>
#     - <a href='#6.1'>6.1 Reducing model</a>
# - <a href='#7'>7. Principal Component analysis</a>
# - <a href='#8'>8. Upsampling</a>
# - <a href='#9'>9. Building classifiers</a>
# - <a href='#10'>10. Classifiers Result</a>
# - <a href='#11'>11. Hypertuning Parameters</a>
# - <a href='#12'>12. Appendix 1 (Step by step analysis)</a>
# - <a href='#13'>13. Appendex 2 (An alternative)</a>
# - <a href='#14'>14. Appendex 3 (CatBoostClassifier)</a>
# - <a href='#15'>15. Appendex 4 (ANN)</a>
# - <a href='#16'>16. Conclusion</a>

# ## Import modules

# In[ ]:


#for data and data visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.graph_objs as go
import plotly.figure_factory as ff
get_ipython().run_line_magic('matplotlib', 'inline')

#Classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Model validation and preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, GridSearchCV, StratifiedShuffleSplit, PredefinedSplit
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

#Deep learning
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

#Helpers
import re
import time
from datetime import datetime
from scipy.stats import boxcox
from collections import Counter
from sklearn_pandas import DataFrameMapper, gen_features
from imblearn.over_sampling import SMOTE
from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # <a id='1'>1. Import and view dataset</a>

# In[ ]:


full = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
full.sample(5)


# ## <a id='1.1'>1.1 Data overview</a>

# In[ ]:


print('Number of customers: {}'.format(full.shape[0]))
print('Number of features: {}'.format(full.shape[1]))
data = full[full.notnull().all(axis=1)].iloc[0].T.to_frame().reset_index()
data.columns = ['Features','Example']
data['Missing values'] = full.isnull().sum().values
data['Number of unique values'] = full.nunique().values
data['Data Type']= full.dtypes.values
data


# Nothing out of the ordinary here except for the data type for TotalCharges, which should be float64. We will take a look into it later.

# # <a id='2'>2. Exploratory analysis</a>

# In[ ]:


##Helper functions
def plot_pie(column):
    churned = full.loc[full['Churn']=='Yes',column].copy()
    notchurned = full.loc[full['Churn']=='No',column].copy()
    pie1 = go.Pie(labels=churned.value_counts().index.tolist(),
                  values=churned.value_counts().values.tolist(),
                  hole=0.5,
                  domain=dict(x=[0,0.48]),
                  hoverinfo='label+value')
    
    pie2 = go.Pie(labels=notchurned.value_counts().index.tolist(),
                  values=notchurned.value_counts().values.tolist(),
                  hole=0.5,
                  domain=dict(x=[0.52,1]), 
                  hoverinfo='label+value')
    
    layout = go.Layout(title='Distribution of {}'.format(column),
                       annotations=[dict(text='Churn',
                                                font=dict(size = 13),
                                                showarrow=False,
                                                x=0.21, y=0.5),
                                           dict(text='Did not churn',
                                                font=dict(size=13),
                                                showarrow=False,
                                                x=0.84,y =0.5
                                               )
                                          ])
    data = [pie1,pie2]
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig)
    
def plot_dist(column):
    churned = full.loc[full['Churn']=='Yes',column].values.tolist()
    notchurned = full.loc[full['Churn']=='No',column].values.tolist()
    
    histdata = [churned,notchurned]
    group_labels = ['Churn','Did not churn']
    fig = ff.create_distplot(histdata,
                             group_labels,
                             show_hist=False,
                             show_rug=False)
    fig['layout'].update(title='Distribution of {}'.format(column),
                         yaxis=dict(title='Distribution probability'),
                         xaxis=dict(showgrid=False,title=column))
    py.iplot(fig)


# ## <a id='2.1'>2.1 Target variable - Churn</a>

# In[ ]:


labels = full['Churn'].value_counts().index.tolist()
values = full['Churn'].value_counts().values.tolist()
pie = go.Pie(labels=labels,values=values,opacity=0.9)
layout = go.Layout(title='Survival rate',
                   autosize=False)
fig = go.Figure(data=[pie],layout=layout)
py.iplot(fig)


# The churn rate is rather imbalanced, with only around 26.5% of the customers churning. This suggest we might have to take some measures before training a model in order to achieve better results. Also, the F1 score evaluation metric would be a more appropriate option than accuracy.

# ## <a id='2.2'>2.2 Gender</a>

# In[ ]:


plot_pie('gender')


# For both churning and non churning customers, the distribution of gender seems fairly similar. This suggest gender would be a good predictive indicator for churning.

# ## <a id='2.3'>2.3 Senior citizen</a>

# In[ ]:


plot_pie('SeniorCitizen')


# From the above pie graphs, it appears that senior citizens are more likely to churn.

# ## <a id='2.4'>2.4 Partner</a>

# In[ ]:


plot_pie('Partner')


# Customers without partners look more likely to churn than those with a partner.

# ## <a id='2.5'>2.5 Dependents</a>

# In[ ]:


plot_pie('Dependents')


# Those customers without dependents appears to be more like to churn those with dependents.

# ## <a id='2.6'>2.6 Tenure</a>

# In[ ]:


plot_dist('tenure')


# Customers with shorter tenures looks more like to churn.

# ## <a id='2.7'>2.7 PhoneService</a>

# In[ ]:


plot_pie('PhoneService')


# From the pie plots, subscribing to phone service does not appear to offer much information for predicting customer churn.

# ## <a id='2.8'>2.8 MultipleLines</a>

# In[ ]:


plot_pie('MultipleLines')


# The similar distribution of both pie graphs suggests having multiple lines or not would not be too important a factor for churning.

# ## <a id='2.9'>2.9 InternetService</a>

# In[ ]:


plot_pie('InternetService')


# The plots suggests that fiber optic users looks very likely to churn

# ## <a id='2.10'>2.10 OnlineSecurity</a>

# In[ ]:


plot_pie('OnlineSecurity')


# From the plots, customers without online security looks very likely to churn.

# ## <a id='2.11'>2.11 OnlineBackup</a>

# In[ ]:


plot_pie('OnlineBackup')


# From the plots, customers without online backup looks very likely to churn.

# ## <a id='2.12'>2.12 DeviceProtection</a>

# In[ ]:


plot_pie('DeviceProtection')


# From the plots, customers without device protection looks very likely to churn.

# ## <a id='2.13'>2.13 TechSupport</a>

# In[ ]:


plot_pie('TechSupport')


# From the plots, customers without technical support looks very likely to churn.

# ## <a id='2.14'>2.14 StreamingTV</a>

# In[ ]:


plot_pie('StreamingTV')


# The plots suggests that those customers without streamingTV are slightly more likely to churn.

# ## <a id='2.15'>2.15 StreamingMovies</a>

# In[ ]:


plot_pie('StreamingMovies')


# The distribution for StreamingTV and StreamingMovies looks very similar. Like streamingTV, those without subscription for streamingMovie are slightly more likely to churn.

# ## <a id='2.16'>2.16 Contract</a>

# In[ ]:


plot_pie('Contract')


# Customers with month-to-month contract seems very likely to churn as compared to those with other forms of other contract.

# ## <a id='2.17'>2.17 PaperlessBilling</a>

# In[ ]:


plot_pie('PaperlessBilling')


# Customers with paperlessbilling looks slightly more likely to churn than those without.

# ## <a id='2.18'>2.18 PaymentMethod</a>

# In[ ]:


plot_pie('PaymentMethod')


# Customers with electronic check as payment method looks more likely to churn.

# ## <a id='2.19'>2.19 MonthlyCharges</a>

# In[ ]:


plot_dist('MonthlyCharges')


# The distribution curves suggest that customers with higher monthly charges are more likely to churn.

# ## <a id='2.20'>2.20 TotalCharges</a>

# As mentioned previously, the data type of 'totalcharges' does not seem right, lets take a look why.

# In[ ]:


full[pd.to_numeric(full['TotalCharges'], errors='coerce').isnull()]


# It seems that for those new customers (tenure being 0), the data assigns an empty string for total charges. Logically, we can assigned the total charges to be 0 for this group of customers. However, taking a look at the target variable, none of these customers churned, which makes sense as new customers would technically be unable to churn. Hence, it is better to drop these customers from the dataset for training purposes, as it is trivial to predict churning for new customers. 

# In[ ]:


full = full[pd.to_numeric(full['TotalCharges'], errors='coerce').notnull()]
print('Number of customers: {}'.format(full.shape[0]))
print('Number of features (including target variable): {}'.format(full.shape[1]))
full.sample(5)


# # <a id='3'>3. Feature engineering</a>

# Feature engineering is a process whereby we create new features based on the available ones to aid the machine learning algorithm. This is usually a way to use one's domain expertise or helping out the machine gather information that is not so obvious from the raw dataset. Afterall, data manipulation in the right way would benefit the machine algorithm greatly later on.

# ## <a id='3.1'>3.1 Ratio of past monthly charges to current month charge</a>

# The logic behind this is that a possible reason for customer churning is due to the increase in charges. 

# In[ ]:


full['ratio'] = (full['TotalCharges'].astype(float).div(full['tenure'])).div(full['MonthlyCharges'])
plot_dist('ratio')


# The distribution looks very similar and might not be too useful. However, we will let the model decide whether it is useful.

# ## <a id='3.2'>3.2 Count of extra services</a>

# Based on the exploratory analysis, we could tell that those customers with additional services are less likely to churn, so we shall just sum the number of extra services the customer signed up for and make it a new feature.

# In[ ]:


full['TotalServices'] = (full[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']]=='Yes').sum(axis=1)
pd.crosstab(index=full['Churn'],columns=full['TotalServices'],normalize='columns')


# It can be observed that the more services the customers subscribed to, the less likely they will churn. The customers with 0 services does not seem to follow the trend but that could be due to the segment of customers including those without internet service.

# In[ ]:


full.sample(5)


# # <a id='4'>4. Split into train and test set</a>

# To validate the models, we will split the dataset provided to train and test set. As the distribution of the target variable is quite skewed, the spliting done would try to ensure similar proportion in both sets. The split for this would be 80% train set and 20% test set.

# In[ ]:


#Seperating the independent and dependent variable
x = full.drop('Churn',axis=1)
y = full['Churn']

#Splitting to train test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=0)

#Check the split
trace1 = go.Table(header=dict(values=[['<b>TRAIN SET</b>'],['<b>COUNTS</b>'],['<b>PERCENTAGE</b>']],
                             fill = dict(color='#C2D4FF'),
                             align = ['left','center']),
                 cells=dict(values=[y_train.value_counts().index.tolist(),
                                    y_train.value_counts().values.tolist(),
                                    np.round(y_train.value_counts(normalize=True).values * 100,2)],
                            align = ['left', 'center'],
                            fill = dict(color='#EDFAFF')),
                domain=dict(x=[0,0.48]))

trace2 = go.Table(header=dict(values=[['<b>TEST SET</b>'],['<b>COUNTS</b>'],['<b>PERCENTAGE</b>']],
                             fill = dict(color='#C2D4FF'),
                             align = ['left','center']),
                 cells=dict(values=[y_test.value_counts().index.tolist(),
                                    y_test.value_counts().values.tolist(),
                                    np.round(y_test.value_counts(normalize=True).values * 100,2)],
                            align = ['left', 'center'],
                            fill = dict(color='#EDFAFF')),
                 domain=dict(x=[0.52,1]))
layout = dict(width=800, height=300)
fig = dict(data=[trace1,trace2],layout=layout)
py.iplot(fig)


# # <a id='5'>5. Data Preprocessing</a>

# As most models available on sklearn requires the features to be numbers, we would have to do some preprocessing in order for them to work. In order to make sure that there is no data leakage, we will fit the transformation on the training set first before transforming the test set.

# In[ ]:


from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#Map churn to integers
y_train = y_train.map({'Yes':1,'No':0})
y_test = y_test.map({'Yes':1,'No':0})

#Dropping customerID as it is just a unique identifier for the customers in the dataset
x_train.drop('customerID',axis=1,inplace=True)
x_test.drop('customerID',axis=1,inplace=True)

#Dropping PhoneService as MultipleLines already provides that information
x_train.drop('PhoneService',axis=1,inplace=True)
x_test.drop('PhoneService',axis=1,inplace=True)

#Replacing 'No internet service' to 'no'
#extra_col = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
#x_train[extra_col] = x_train[extra_col].replace('No internet service','No')
#x_test[extra_col] = x_test[extra_col].replace('No internet service','No')

#Convert binary features to 1 and 0 (avoids unnecessary increase in dimensionality) 
bi_col = ['gender','Partner','Dependents','PaperlessBilling']
feature_LE = gen_features([[i,] for i in bi_col],[LabelEncoder])
mapper_LE = DataFrameMapper(feature_LE,df_out=True,default=None)
x_train = mapper_LE.fit_transform(x_train)
x_test = mapper_LE.transform(x_test)

#One hot encode categorical columns with more than 2 categories
multi_col = ['MultipleLines','InternetService','Contract','PaymentMethod','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
x_train = pd.get_dummies(x_train,columns=multi_col)
x_test = pd.get_dummies(x_test,columns=multi_col)

#Drop 'No internet service' as the information is duplicated many times
extra_col = ['OnlineSecurity_No internet service','OnlineBackup_No internet service','DeviceProtection_No internet service','TechSupport_No internet service','StreamingTV_No internet service','StreamingMovies_No internet service']
x_train.drop(extra_col,axis=1,inplace=True)
x_test.drop(extra_col,axis=1,inplace=True)

#StandardScalar numerical columns 
num_col = ['tenure','MonthlyCharges','TotalCharges','TotalServices']
feature_SS = gen_features([[i] for i in num_col],[StandardScaler])
mapper_SS = DataFrameMapper(feature_SS,df_out=True,default=None)
x_train = mapper_SS.fit_transform(x_train)
x_test = mapper_SS.transform(x_test)

#Convert dataframe to numeric instead of objects to save memory space
x_train = x_train.apply(pd.to_numeric)
x_test = x_test.apply(pd.to_numeric)

#View preprocessed data
temp = x_train.head()
temp['Churn'] = y_train.head()
print('Number of features (including target variable): {}'.format(temp.shape[1]))
temp


# # <a id='6'>6. Feature importance (Selection)</a>

# Feature importance is a way to tell us which feature offer the most infomation on the target variable. It is also a good way to reduce noise in our model by selecting only the top few features, hence it can also be called feature selection.

# In[ ]:


model = RandomForestClassifier(n_estimators=1000,random_state=0)
model = model.fit(x_train, y_train)
features = pd.DataFrame(index=x_train.columns)
features['Importance'] = model.feature_importances_
features = features.sort_values('Importance',ascending=False)
#Visualing feature importance
data= [go.Bar(
    x=features.values.flatten()[::-1],
    y=features.index[::-1],
    orientation='h',
    opacity=0.8)]

layout = go.Layout(title='Feature Importance',
                   autosize=True,
                   xaxis=dict(title='Features',tickangle=0,fixedrange=True),
                   yaxis=dict(fixedrange=True,tickangle=0,tickfont=dict(size=6)),
                   margin=dict(l=200,t=0))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## <a id='6.1'>6.1 Reducing model</a>

# Now that we know which features offers the most explanation, we shall select the top 20 features. This could potentially improve the models as there would be less noise.

# In[ ]:


#Select top k features
top_k = 20
selected_features = features.index.tolist()[:top_k]
x_train_reduced = x_train[selected_features]
x_test_reduced = x_test[selected_features]
x_train_reduced.head()


# # <a id='7'>7. Principal Component analysis (Extraction)</a>

# Another method of dimensionality reduction is using feature extraction. The difference between feature extraction and selection is that while selection chooses a subset of all the features, feature extraction creates new features. One popular method for feature extraction is principal component analysis. PCA works by extracting linear composites of the features, to maximize total variance to bring out distinguishable pattern in the dataset.
# 
# Before transforming the dataset using PCA, we have to decide how many components we want. This can be done by looking at the total variance ratio.

# In[ ]:


pca = PCA(n_components = None)
pca.fit(x_train_reduced)
explained_variance = pca.explained_variance_ratio_

#Bar plot
trace1 = go.Bar(x=[i for i in range(1,21)],
                y=explained_variance)

#Graph
trace2 = go.Scatter(x = [i for i in range(1,21)],
                    y = np.cumsum(explained_variance),
                    mode = 'lines+markers'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Variation','Cumulated Variation'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout']['xaxis1'].update(dtick=1)
fig['layout']['xaxis2'].update(dtick=1,showgrid=False)
fig['layout']['yaxis2'].update(dtick=0.05)
fig['layout'].update(showlegend=False)
clear_output(wait=True)
py.iplot(fig)


# Having 10 components explain 90% of the total variance, which would be a good choice.

# In[ ]:


n_components=10
pca = PCA(n_components=n_components)
colnames = ['PC{}'.format(i+1) for i in range(n_components)]
x_train_pca = pd.DataFrame(pca.fit_transform(x_train_reduced, y_train),index=x_train.index,columns=colnames)
x_test_pca = pd.DataFrame(pca.transform(x_test_reduced),index=x_test.index,columns=colnames)
x_train_pca.head()


# # <a id='8'>8. Upsampling</a>

# Upsampling would allow a better performance for the classifiers later as the distribution of the target variable would be more balanced. We would be using SMOTE provided by sklearn to upsample. SMOTE creates synthetic observations of the minority class (churners) by:
# 1. Finding the k-nearest-neighbors for minority class observations (finding similar observations)
# 2. Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.
# 
# We would upsample the minority class till 0.5 of the majority class.

# In[ ]:


sm = SMOTE(random_state=0,ratio=0.5)
x_train_sm, y_train_sm = sm.fit_sample(x_train_pca, y_train)
x_train_sm = pd.DataFrame(x_train_sm,columns=x_train_pca.columns)
y_train_sm = pd.DataFrame(y_train_sm)


# In[ ]:


counter = Counter(y_train_sm.values.flatten())
print('Number of non churners: {}\nNumber of churners: {}'.format(counter[0],counter[1]))


# # <a id='9'>9. Building classifiers</a>

# With all the data manipulation done, we can proceed to the fun part, fitting the data to some classifier models. The models choosen are some of the more common ones like LogisticRegression, SVC,  XGB, LGB.

# In[ ]:


classifiers = {'LogReg': LogisticRegression(),
               'RidgeClassifier': RidgeClassifierCV(),
               'KNN': KNeighborsClassifier(),
               'SVC': SVC(gamma='auto'),
               'GaussianNB': GaussianNB(priors=[0.27, 0.73]),
               'DecisionTree': DecisionTreeClassifier(),
               'RandomForest': RandomForestClassifier(n_estimators=100),
               'AdaBoost': AdaBoostClassifier(n_estimators=100),
               'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
               'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
               'BaggingClassifier': BaggingClassifier(n_estimators=100),
               'XGB': XGBClassifier(),
               'LDA': LinearDiscriminantAnalysis(),
               'LGB': LGBMClassifier()}


# To determine which is the best model, we would do a cross validation and look at the F1 scores. The splitting done would be 80-20 using stratifiedshufflesplit by sklearn.

# In[ ]:


scoring = {'accuracy' : make_scorer(accuracy_score), 
           'f1_score' : make_scorer(f1_score),
           'recall' : make_scorer(recall_score),
           'precision' : make_scorer(precision_score),
           'AUC' : make_scorer(roc_auc_score)}

cv_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=0)
selection_cols = ['Classifier','Mean Train Accuracy','Mean Test Accuracy','Mean F1 train','Mean F1 Test','Mean Train Recall','Mean Test Recall','Mean Train Precision','Mean Test Precision','Mean Train AUC','Mean Test AUC']

classifiers_summary = pd.DataFrame(columns=selection_cols)
start_time = time.time()

for name,classifier in classifiers.items():
    print('Validating ',name)
    clear_output(wait=True)
    cv = cross_validate(classifier,x_train_sm,y_train_sm,return_train_score=True,cv=cv_split,scoring=scoring)
    cv_calc = [name,
               cv['train_accuracy'].mean(),
               cv['test_accuracy'].mean(),
               cv['train_f1_score'].mean(),
               cv['test_f1_score'].mean(),
               cv['train_recall'].mean(),
               cv['test_recall'].mean(),
               cv['train_precision'].mean(),
               cv['test_precision'].mean(),
               cv['train_AUC'].mean(),
               cv['test_AUC'].mean(),
              ]
    cv_calc_s = pd.Series(cv_calc,index=selection_cols)
    classifiers_summary = classifiers_summary.append(cv_calc_s,ignore_index=True)
  
clear_output(wait=True)
run_min,run_sec = divmod(time.time()-start_time,60)
print('Total validating time: {:.0f}min {:.0f}sec'.format(run_min,run_sec))


# In[ ]:


classifiers_summary = classifiers_summary.sort_values('Mean F1 Test',ascending=False)
classifiers_summary_styled = classifiers_summary[selection_cols].style.highlight_max(axis=0).set_properties(**{'width': '150px'})
classifiers_summary_styled


# Looking at the table (sorted descending by F1 test score), randomforest looks to be the best classifiers. However, looking at the train scores, it seems that there could be a possibility of overfitting. Using that train of thought, LDA, RidgeClassifier, GradientBoosting, XGB and LogReg looks to be the top 5 models. To make sure, we can take a look at the scores on the test set. 

# # <a id='10'>10. Classifiers results</a>

# In[ ]:


classifiers = {'LogReg': LogisticRegression(),
               'RidgeClassifier': RidgeClassifierCV(),
               'KNN': KNeighborsClassifier(),
               'SVC': SVC(gamma='auto'),
               'GaussianNB': GaussianNB(priors=[0.27,0.73]),
               'DecisionTree': DecisionTreeClassifier(),
               'RandomForest': RandomForestClassifier(n_estimators=100),
               'AdaBoost': AdaBoostClassifier(n_estimators=100),
               'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
               'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
               'BaggingClassifier': BaggingClassifier(n_estimators=100),
               'XGB': XGBClassifier(),
               'LDA': LinearDiscriminantAnalysis(),
               'LGB': LGBMClassifier()}

selection_cols = ['Classifier',
                  'Accuracy',
                  'F1',
                  'Precision',
                  'Recall',
                  'AUC']

classifiers_summary = pd.DataFrame(columns=selection_cols)
predictions = pd.DataFrame({'Actual':y_test})
start_time = time.time()

for name,classifier in classifiers.items():
    print('Validating ',name)
    clear_output(wait=True)
    classifier.fit(x_train_sm, y_train_sm)
    pred = classifier.predict(x_test_pca)
    cv_calc = [name,
               accuracy_score(y_test,pred),
               f1_score(y_test,pred),
               precision_score(y_test,pred),
               recall_score(y_test,pred),
               roc_auc_score(y_test,pred)
              ]
    predictions[name] = pred
    cv_calc_s = pd.Series(cv_calc,index=selection_cols)
    classifiers_summary = classifiers_summary.append(cv_calc_s,ignore_index=True)
 
clear_output(wait=True)
run_min,run_sec = divmod(time.time()-start_time,60)
print('Total validating time: {:.0f}min {:.0f}sec'.format(run_min,run_sec))


# In[ ]:


classifiers_summary = classifiers_summary.sort_values('F1',ascending=False)

classifiers_summary_styled = classifiers_summary[selection_cols].style.highlight_max(axis=0).set_properties(**{'width': '150px'})
classifiers_summary_styled

#Comparison visualization
trace1 = go.Bar(x=classifiers_summary['Accuracy'].values.tolist()[::-1],
                y=classifiers_summary['Classifier'].values.tolist()[::-1],
                name='Accuracy',
                marker=dict(color='red'),
                orientation='h',
                opacity=0.7)
    
trace2 = go.Bar(x=classifiers_summary['F1'].values.tolist()[::-1],
                y=classifiers_summary['Classifier'].values.tolist()[::-1],
                name='F1',
                marker=dict(color='blue'),
                orientation='h',
                opacity=0.6)

trace3 = go.Bar(x=classifiers_summary['Precision'].values.tolist()[::-1],
                y=classifiers_summary['Classifier'].values.tolist()[::-1],
                name='Precision',
                marker=dict(color='green'),
                orientation='h',
                opacity=0.6)

trace4 = go.Bar(x=classifiers_summary['Recall'].values.tolist()[::-1],
                y=classifiers_summary['Classifier'].values.tolist()[::-1],
                name='Recall',
                marker=dict(color='blueviolet'),
                orientation='h',
                opacity=0.6)

trace5 = go.Bar(x=classifiers_summary['AUC'].values.tolist()[::-1],
                y=classifiers_summary['Classifier'].values.tolist()[::-1],
                name='AUC',
                marker=dict(color='gray'),
                orientation='h',
                opacity=0.6)

data = [trace1,trace2,trace3,trace4,trace5]
layout = go.Layout(title='Scoring of classifiers',
                   autosize=True,
                   xaxis=dict(title='Scores',tickangle=0,fixedrange=True,range=[0,0.9],dtick=0.05),
                   yaxis=dict(fixedrange=True,tickangle=-30))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


fig = plt.figure(figsize=(30,17.5))
sns.set(font_scale=1.5)
for i,c in enumerate(classifiers_summary['Classifier'][::-1]):
    plt.subplot(3,5,i+1)
    sns.heatmap(pd.crosstab(y_test,predictions[c]).values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
    plt.title(c,fontdict=dict(fontweight='bold'))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    clear_output(wait=True)
plt.tight_layout()


# As expected, random forest has overfitted on the training set. XGB looks to be the best based on F1 score, however there are models such as LDA that predicts more churners accurately. In the end, you have to choose between and balance of recall and precision to make your choice.

# # <a id='12'>12. Appendix 1 (Step by step analysis)</a>

# Lets take a look at how the data manipulation improved the scores at each step. We would just use the default XGBoost classifier to test out the changes.

# In[ ]:


#Without feature engineering
temp = XGBClassifier()
temp.fit(x_train.drop(['TotalServices','ratio'],axis=1),y_train)
cm_no_fe = pd.crosstab(y_test,temp.predict(x_test.drop(['TotalServices','ratio'],axis=1)))

#No dimension reduction
temp = XGBClassifier()
temp.fit(x_train,y_train)
cm_no_dr = pd.crosstab(y_test,temp.predict(x_test))

#Just top 20 features
temp = XGBClassifier()
temp.fit(x_train_reduced,y_train)
cm_no_pca = pd.crosstab(y_test,temp.predict(x_test_reduced))

#Without upsampling
temp = XGBClassifier()
temp.fit(x_train_pca,y_train)
cm_no_us = pd.crosstab(y_test,temp.predict(x_test_pca))

clear_output()


# In[ ]:


from matplotlib.gridspec import GridSpec
fig=plt.figure(figsize=(10,10))
gs=GridSpec(4,4) 

#Base model
ax1=fig.add_subplot(gs[0:4,0:3]) 
sns.set(font_scale=2)
sns.heatmap(pd.crosstab(predictions['Actual'],predictions[c]).values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
ax1.title.set_text('Base model')
ax1.title.set_fontweight('bold')

#No feature engineering
sns.set(font_scale=1)
ax2=fig.add_subplot(gs[0,3]) 
sns.heatmap(cm_no_fe.values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
ax2.title.set_text('No feature engineering')
ax2.title.set_fontweight('bold')

#No dimension reduction
ax3=fig.add_subplot(gs[1,3])
sns.heatmap(cm_no_dr.values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
ax3.title.set_text('No dimension reduction')
ax3.title.set_fontweight('bold')

#No PCA
ax4=fig.add_subplot(gs[2,3]) 
sns.heatmap(cm_no_pca.values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
ax4.title.set_text('No PCA')
ax4.title.set_fontweight('bold')

#No upsampling
ax5=fig.add_subplot(gs[3,3]) 
sns.heatmap(cm_no_us.values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
ax5.title.set_text('No Upsampling')
ax5.title.set_fontweight('bold')
clear_output(wait=True)
fig.tight_layout()


# It does seem that the data manipulation does provide better information for the model, however the model performs better without dimension reduction. In that case lets try upsampling without dimension reduction.

# In[ ]:


#Upsampling without dimension reduction
sm = SMOTE(random_state=0,ratio=0.5)
x_train_sm2, y_train_sm2 = sm.fit_sample(x_train, y_train)
x_train_sm2 = pd.DataFrame(x_train_sm2,columns=x_train.columns)
y_train_sm2 = pd.DataFrame(y_train_sm2)

#Fit XGB and predict
temp = XGBClassifier()
temp.fit(x_train_sm2,y_train_sm2)
sns.heatmap(pd.crosstab(y_test,temp.predict(x_test)).values,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='jet',
                xticklabels=['No churn','Churn'],
                yticklabels=['No churn','Churn'],
                linewidth=2
               )
clear_output(wait=True)


# While the result without upsampling was slightly better without upsampling, it was the other way around with upsampling.

# # <a id='13'>13. Appendix 2 (An alternative)</a>

# Instead of upsampling, we can reduce, or increase depending on which values you are looking at, the probability to determine whether the customer churns or not. By default, for binary classification like this dataset, the magic number is 0.5, we can reduce the number to get a better result.

# In[ ]:


temp = XGBClassifier()
temp.fit(x_train_pca,y_train)
confidence_level = 0.432
preds = np.where(temp.predict_proba(x_test_pca)[:,1]>confidence_level,1,0)
sns.heatmap(pd.crosstab(y_test,preds).values,
            annot=True,
            fmt='d',
            cbar=False,
            cmap='jet',
            xticklabels=['No churn','Churn'],
            yticklabels=['No churn','Churn'],
            linewidth=2
           )
plt.title('Confidence level: {}'.format(confidence_level),fontdict=dict(fontweight='bold'))
clear_output(wait=True)


# # <a id='14'>14. Appendix 3 (CatBoostClassifier)</a>

# While most models provided by sklearn require the features to be numerical, catboost does not and in a sense, it's sort of a short cut as we do not really have to preprocess the data for the classifier to work. We try shall try it out as an example. The only data manipulation we would be doing is to remove those with no 'TotalCharges' and the 2 feature engineering to keep things consistent. (Please note that the parameters are manually optimize as much as possible to save time.) 

# In[ ]:


get_ipython().run_cell_magic('capture', '', "from catboost import CatBoostClassifier\n#Reset data\nfull = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')\n\n#Remove blank totalcharges\nfull = full[full['TotalCharges']!=' ']\n#full['TotalCharges'] = full['TotalCharges'].astype(float)\nfull = full.apply(pd.to_numeric, errors='ignore')\n\n#Feature engineering\nfull['ratio'] = (full['TotalCharges'].astype(float).div(full['tenure'])).div(full['MonthlyCharges'])\nfull['TotalServices'] = (full[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']]=='Yes').sum(axis=1)\n\n#Split to train test set\nx = full.drop(['customerID','Churn'],axis=1)\ny = full['Churn'].map({'Yes':1,'No':0})\nx_train_cb,x_test_cb,y_train_cb,y_test_cb = train_test_split(x,y,test_size=0.2,stratify=y,random_state=0)\n\n#Train catboost classifier\ncat_features =  [i for i,e in enumerate(x_train_cb.columns) if not np.issubdtype(x_train_cb[e].dtype , np.number)]\nmodel=CatBoostClassifier(iterations=200, depth=2, learning_rate=0.5,eval_metric='F1',random_seed=0)\nmodel.fit(x_train_cb,\n          y_train_cb,\n          cat_features=cat_features,\n          eval_set=(x_test_cb, y_test_cb),\n          early_stopping_rounds=100,\n          #plot=True,\n          verbose = 50,\n          use_best_model=True)")


# In[ ]:


fig=plt.figure(figsize=(10,6))
gs=GridSpec(1,2) 

#0.5
ax1=fig.add_subplot(gs[0,0]) 
cat_pred = model.predict(x_test_cb)
sns.heatmap(pd.crosstab(y_test_cb, cat_pred).values,
            annot=True,
            fmt='d',
            cbar=False,
            cmap='jet',
            xticklabels=['No churn','Churn'],
            yticklabels=['No churn','Churn'],
            linewidth=2
           )
ax1.title.set_text('0.5')
ax1.title.set_fontweight('bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#Changing probability
ax2=fig.add_subplot(gs[0,1])
confidence_level = 0.433
cat_pred2 = np.where(model.predict_proba(x_test_cb)[:,1]>confidence_level,1,0)
sns.heatmap(pd.crosstab(y_test_cb, cat_pred2).values,
            annot=True,
            fmt='d',
            cbar=False,
            cmap='jet',
            xticklabels=['No churn','Churn'],
            yticklabels=['No churn','Churn'],
            linewidth=2
           )
ax2.title.set_text(confidence_level)
ax2.title.set_fontweight('bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
clear_output(wait=True)
plt.tight_layout()


# It appears that the CatBoost classifier performs almost as well as our choosen classifier, XGB, with only an additional 5 false positives and the same number of true positives. (Catboost classifier was manually optimized as much as possible and changing the confidence required to predict was a way around upsampling. Upsampling using SMOTE was troublesome for CatBoost as SMOTE required numerical data, which defeats the purpose of showing CatBoost.) Looking at the length of the codes, maybe trading 5 extra errors would be worth it to some.

# # <a id='15'>15. Appendix 4 (ANN)</a>

# Deep learning is a very popular machine tool in modern data science. We shall attempt using a simple model to see how well it does. However, it is important to note that the dataset is rather small and deep learning might not be as effective.

# In[ ]:


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_f1', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# ## ANN structure

# In[ ]:


ann = Sequential()
ann.add(Dense(units=15, kernel_initializer='RandomUniform', activation='relu', input_dim=10))
ann.add(Dense(units=10, kernel_initializer='RandomUniform', activation='relu'))
ann.add(Dense(units=1, kernel_initializer='RandomUniform', activation='sigmoid'))
ann.compile(optimizer=Adam(lr=.01), loss='binary_crossentropy', metrics=[f1])
ann.summary()


# ## Training the ANN

# In[ ]:


epochs = 30
callback = ann.fit(x_train_sm,y_train_sm,validation_data=(x_test_pca,y_test),epochs=epochs,batch_size=48,verbose=2,callbacks=[learning_rate_reduction])


# In[ ]:


#Loss
trace1 = go.Scatter(x=[i for i in range(1,epochs+1)],
                    y=callback.history['val_loss'],
                    marker = dict(color='red'),
                    mode = 'lines+markers',
                    name = 'Validation'
                   )

trace2 = go.Scatter(x=[i for i in range(1,epochs+1)],
                    y=callback.history['loss'],
                    marker = dict(color='blue'),
                    mode = 'lines+markers',
                    name = 'Train'
                   )
#F1
trace3 = go.Scatter(x = [i for i in range(1,epochs+1)],
                    y = callback.history['val_f1'],
                    marker = dict(color='red'),
                    mode = 'lines+markers',
                    name = 'Validation'
                   )

trace4 = go.Scatter(x = [i for i in range(1,epochs+1)],
                    y = callback.history['f1'],
                    marker = dict(color='blue'),
                    mode = 'lines+markers',
                    name = 'Train'
                   )

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Loss','F1 Score'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout']['xaxis1'].update(dtick=1,showgrid=False,tickangle=0)
fig['layout']['xaxis2'].update(dtick=1,showgrid=False,tickangle=0)
fig['layout']['yaxis2'].update(dtick=0.05)
fig['layout'].update(showlegend=True)
clear_output(wait=True)
py.iplot(fig)


# The graphs suggest that the artificial neural network did not overfit but there was no significant improvements to the predictions after the 10th epoch.

# In[ ]:


ann_pred = ann.predict(x_test_pca).flatten()
ann_pred = (ann_pred>0.5).astype(int)
sns.heatmap(pd.crosstab(y_test, ann_pred).values,
            annot=True,
            fmt='d',
            cbar=False,
            cmap='jet',
            xticklabels=['No churn','Churn'],
            yticklabels=['No churn','Churn'],
            linewidth=2
           )
plt.title('ANN',fontdict=dict(fontweight='bold'))
plt.xlabel('Predicted')
plt.ylabel('Actual')
clear_output(wait=True)


# The ANN performs decently but still loses out to the other top performers. 

# # <a id='16'>16. Conclusion</a>

# This notebook has shown some ways to manipulate data, some methods to select models, and how to hypertune parameters. Overall, the results are generally acceptable (though not fantastic) with most models having around 80% accuracy and slightly above 60% F1 score. 
