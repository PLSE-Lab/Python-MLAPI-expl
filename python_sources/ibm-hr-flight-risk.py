#!/usr/bin/env python
# coding: utf-8

# This is my first kernel in Kaggle and I am going to use this kernel as a start for getting to gain knowledge in Kaggle

# ****Into to Data****
# 
# First let us try to understand what is employee attrition.
# Attrition is a term used most often in the case when an employee resigns or retires, sometimes they are also called employee churn which again means losing someone from a company. In this case we are trying to predict employees who will resign voluntarily and not involuntarily(also called layoff,this depends on the budget for the team and many other factors)
# 
# ****Why is predicting attrition important****
# 
# Attrition affects a lot of factors in an organization and some of them are as follows:
# 
# 1. A lot of cost is involved when an employee is hired, trained and during knowledge transfer in the new team. So when the employee leaves in 6 months or 1 year the entire cost spent on this is lost.
# 2. To replace the employee who has quit there again needs to be money spent on job posting, recruiting and knowledge transfer.
# 3. Meanwhile the team who has lost the team member will suffer due to shortage of resources and the team's productivity will go down.
# 4. The task handled by the employee who quit will have to be managed by existing team member on top of his/her regular task, making his/her life a pain.
# 5. Often a job role takes atleast 3 to 4 months to be filled and sometimes more if the candidate is on work visa and needs his visa transferred. Also not to mention the extra visa costs will be added to the department.
# 
# So according to a research the total attrition cost which involves hiring, job posting, training and knowledge transfer will all amount to the annual salary of the person who quit and in some cases 2x annual salary if the skills sets are rare.

# **Now lets get started with our Analysis**

# In[ ]:


from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from statsmodels.stats.outliers_influence import variance_inflation_factor  


from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import train_test_split
from matplotlib.ticker import StrMethodFormatter

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge


from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn import neighbors
from datetime import datetime
from sklearn import ensemble
from scipy.stats import norm
from sklearn import metrics
from scipy import stats
import lightgbm as lgb
import seaborn as sns
import xgboost as xgb

import numpy as np 
import pandas as pd 
import seaborn as sns
import pandas_profiling
from pathlib import Path

get_ipython().run_line_magic('matplotlib', 'inline')

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re


import h2o
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from h2o.automl import H2OAutoML
h2o.init()


# Now that all the packages are loaded. Let's define some functions like Model performance plot, Feature importance, ROC/AUC curve and cross validation.
# 
# *****These modules & some coding were taken from another Kernel, Thank you to the OP!!*****

# In[ ]:


def model_performance_plot(model) : 
    #conf matrix
    conf_matrix = confusion_matrix(valid_y, y_pred)
    trace1 = go.Heatmap(z = conf_matrix  ,x = ["0 (pred)","1 (pred)"],
                        y = ["0 (true)","1 (true)"],xgap = 2, ygap = 2, 
                        colorscale = 'Viridis', showscale  = False)

    #show metrics
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))
    Precision =  (tp/(tp+fp))
    Recall    =  (tp/(tp+fn))
    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x = (show_metrics[0].values), 
                   y = ['Accuracy', 'Precision', 'Recall', 'F1_score'], text = np.round_(show_metrics[0].values,4),
                    textposition = 'auto',
                   orientation = 'h', opacity = 0.8,marker=dict(
            color=colors,
            line=dict(color='#000000',width=1.5)))
    
    #plot roc curve
    model_roc_auc = round(roc_auc_score(valid_y, y_score) , 3)
    fpr, tpr, t = roc_curve(valid_y, y_score)
    trace3 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : ",
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2), fill='tozeroy')
    trace4 = go.Scatter(x = [0,1],y = [0,1],
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))
    
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(valid_y, y_score)
    trace5 = go.Scatter(x = recall, y = precision,
                        name = "Precision" + str(precision),
                        line = dict(color = ('lightcoral'),width = 2), fill='tozeroy')
    
    #subplots
    fig = tls.make_subplots(rows=2, cols=2, print_grid=False, 
                        subplot_titles=('Confusion Matrix',
                                        'Metrics',
                                        'ROC curve'+" "+ '('+ str(model_roc_auc)+')',
                                        'Precision - Recall curve'))
    
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,2,1)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5,2,2)
    
    fig['layout'].update(showlegend = False, title = '<b>Model performance</b><br>'+str(model),
                        autosize = False, height = 900,width = 830,
                        plot_bgcolor = 'rgba(240,240,240, 0.95)',
                        paper_bgcolor = 'rgba(240,240,240, 0.95)',
                        margin = dict(b = 195))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis3"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis4"].update(dict(title = "recall"), range = [0,1.05])
    fig["layout"]["yaxis4"].update(dict(title = "precision"), range = [0,1.05])
    fig.layout.titlefont.size = 14
    
    py.iplot(fig)


# In[ ]:


def features_imp(model, cf) : 

    coefficients  = pd.DataFrame(model.feature_importances_)
    column_data     = pd.DataFrame(list(data))
    coef_sumry    = (pd.merge(coefficients,column_data,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    coef_sumry = coef_sumry[coef_sumry["coefficients"] !=0]
    trace = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Viridis",
                                  line = dict(width = .6,color = "black")))
    layout = dict(title =  'Feature Importances xgb_cfl')
                    
    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)


# In[ ]:


#cumulative gain curve
def cum_gains_curve(model):
    pos = pd.get_dummies(y_test).as_matrix()
    pos = pos[:,1] 
    npos = np.sum(pos)
    index = np.argsort(y_score) 
    index = index[::-1] 
    sort_pos = pos[index]
    #cumulative sum
    cpos = np.cumsum(sort_pos) 
    #recall
    recall = cpos/npos 
    #size obs test
    n = y_test.shape[0] 
    size = np.arange(start=1,stop=369,step=1) 
    #proportion
    size = size / n 
    #plots
    model = 'xgb_cfl'
    trace1 = go.Scatter(x = size,y = recall,
                        name = "Lift curve",
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    trace2 = go.Scatter(x = size,y = size,
                        name = "Baseline",
                        showlegend=False,
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))

    layout = dict(title = 'Cumulative gains curve'+' '+str(model),
                  yaxis = dict(title = 'Percentage positive targeted',zeroline = False),
                  xaxis = dict(title = 'Percentage contacted', zeroline = False)
                 )

    fig  = go.Figure(data = [trace1,trace2], layout = layout)
    py.iplot(fig)


# In[ ]:


# Cross val metric
def cross_val_metrics(model) :
    scores = ['accuracy', 'precision', 'recall']
    for sc in scores:
        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)
        print('[%s] : %0.5f (+/- %0.5f)'%(sc, scores.mean(), scores.std()))


# We are going to use pandas to load our csv source file into a dataframe

# In[ ]:


df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv', encoding = "ISO-8859-1",
                 low_memory = False)

# Let's look at a sample of the dataset to understand the format and type of features
df.sample(5)


# To get a good understanding of the data we must look at a descriptive report of the data.
# 
# We can use the pandas profiling for this purpose which gives a detailed analyis  of the data

# In[ ]:


pandas_profiling.ProfileReport(df)


# In[ ]:


print('We have %d features' %len(df.columns))

print('The name of the features are as below :')

print(df.columns)


# In[ ]:


df = df.drop(['EmployeeCount','Over18','StandardHours'], axis = 1)


# In[ ]:


cols = ['DailyRate','DistanceFromHome','Age','HourlyRate','MonthlyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany']

n_bins = 7
for i in cols:
    lower, higher = df[i].min(), df[i].max()
    edges = range(lower, higher, round((higher - lower)/n_bins))
    lbs = ['(%d, %d]'%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    df[i] = pd.cut( x = df[i], bins=edges, labels=lbs, include_lowest=True)


# In[ ]:


df.head(5)


# We are going to be doing some basic feature engineering

# Below is a plot to understand the distribution of the data

# In[ ]:


total_records= len(df)
columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']
plt.figure(figsize=(20,20))
j=0
for i in columns:
    j +=1
    plt.subplot(19,2,j)
    #sns.countplot(hrdata[i])
    ax1 = sns.countplot(data=df,x= i,hue="Attrition")
    if(j==8 or j== 7):
        plt.xticks( rotation=90)
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}'.format(height,0),
                ha="center",rotation=0) 

# Custom the subplot layout
plt.subplots_adjust(bottom=-0.9, top=2)
plt.show()


# In[ ]:


#Let us look at the correlation between the numerical variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
numerical = df.select_dtypes(include = numerics)
numerical.sample(5)


# In[ ]:


# Correlation between variables helps us understand which of these variables could make a change in other variables and we could remove one of them if their correlation coefficient is more than a threshold we could specify
# we are using spearman instead of Pearson is because our data is related to human behavior and it is not linear

get_ipython().run_line_magic('matplotlib', 'notebook')
import plotly.figure_factory as ff

dfcorr = numerical.corr(method = 'spearman')
num_cols = dfcorr.columns.tolist()
num_array = np.array(dfcorr)

# Let us plot the data

trace = go.Heatmap(z= num_array,
                   x = num_cols,
                   y = num_cols,
                   colorscale = 'Electric',
                   colorbar = dict(),)

layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        height  = 1400,
                        width   = 1600,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )

fig = go.Figure(data=[trace], layout = layout)
py.iplot(fig)


# In[ ]:


# Remove highly correlated variables

threshold = 0.85

datacorrs = numerical.corr(method ='spearman').abs()

upper = datacorrs.where(np.triu(np.ones(datacorrs.shape),k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any (upper[column] >= threshold)]

df = df.drop(columns = to_drop)
print(to_drop)


# In[ ]:


# Let's see how the distribution of data when it comes to active and inactive employees
data = [go.Bar(
            x=df["Attrition"].value_counts().index.values,
            y= df["Attrition"].value_counts().values
    )]

py.iplot(data, filename='basic-bar')


# In[ ]:


# Let us start with our Target variable which is attrition and it is binary, So let us convert attrition = yes to 1 and attrition = No to 0

target_map = {'Yes':1, 'No':0}
df['Attrition'] = df['Attrition'].apply(lambda x : target_map[x])

# Store the target variable Attrition separate and remove it from the dataset

target = df['Attrition']
df = df.drop('Attrition', 1)
df = df.drop('EmployeeNumber', 1)

# Now begins the one hot encoding for the remaining variables
category = []
for cols, val in df.iteritems():
    if val.dtype !='int64':
        category.append(cols)
        

numeric = df.columns.difference(category)

df_categor = df[category]
df_cat = pd.get_dummies(df_categor)
df_num = df[numeric]
df = pd.concat([df_num, df_cat, target], axis = 1)
df.sample(5)


# In[ ]:


data = df.copy()
# Shuffle the dataset
df.sample(frac=0.1)
# Split them into train and test
traindata = df.sample(frac=0.8, random_state = 200)
testdata = df.drop(traindata.index)
# Converting the datasets into H2o frame
train_data = h2o.H2OFrame(traindata)
test_data = h2o.H2OFrame(testdata)
x = train_data.columns
y = "Attrition"
x.remove(y)
# For binary classification, response should be factor
train_data[y] = train_data[y].asfactor()
test_data[y] = test_data[y].asfactor()
# Let the game begin
aml = H2OAutoML(max_models=5, seed=100, nfolds = 5, max_runtime_secs = 200, max_runtime_secs_per_model = 240)
aml.train(x=x, y=y , training_frame = train_data)


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# **VOILA!!! We have an accuracy of 83%**
