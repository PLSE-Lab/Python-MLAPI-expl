#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook will pay attention to retention (of employees)! 
# 
# For almost any company, employees leaving (aka attrition) can have significant impact and cost. For one, the employee may have a depth of experience and talent that is vital to the company (or a team). In addition, replacing such an employee may be difficult (if not nearly impossible). For another, it takes resources and time to bring on a new employee to 'replace' the talent/expertise/knowledge of the former employee. 
# 
# With that being said, this notebook will use employee (attrition) data provided by IBM data scientists. 
# 
# Exploratory data analysis will be performed to look at the data and visualize certain aspects of it. During EDA, some preprocessing will be done to the initial data set in order to prepare it for some machine learning techniques. 
# 
# Then, classification models like Logistic Regression, kNN, LightGBM, SVC, Neural Nets, etc. will be used to help predict which employees are at risk of leaving. The classification models and techniques will be compared with each other to assess performance against each other. 

# # Data
# 
# The data contains information on 1470 employees. There are 35 features to each employee with one being the unique identifier (employee ID number). Another important feature is attrition which contains 'Yes' or 'No' for wether the employee left or not. This will be our target variable (or y). 
# 
# It should be noted that this is a fictional data set created by IBM data scientists.

# In[ ]:


# Import needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Import Models
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Define constants/variables to use
seed_Num = 8 # aka Random State
num_folds = 5


# In[ ]:


# Read in the data into pandas dataframe
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


pd.DataFrame(data.columns, columns=['Column Names'])


# # Exploratory Data Analysis (and Cleaning Data)
# 
# This section will go into some exploratory data analysis along with cleaning the data for further analysis. 
# 
# Let us now take a look at a few random rows of data to get a sense of what the data looks like.
# 

# In[ ]:


data.sample(5, random_state=8)


# The leftmost column of numbers next to Age represent the index of the row in the data (so we have rows 1441, 81, 243, 1142, and 51). Remember, each row represents an employee. As an example to understand what a row is telling us, reading the first few features (columns) of the 1441 row represents an employee who is 56 years old (Age), has not left the company (Attrition), does not travel (BusinessTravel), gets paid 667 a day (DailyRate), and is part of the research & development department (Department).
# 
# Just looking at all the columns for these few rows, some variables/features have the same values, like department, employee count, number of companies worked, over 18, and standard hours. Features with the same value for all employees do not help us with predicting attrition (but can potentially give other information). Also, features with a unique value for each sample/employee, like employee ID number, will not be used for helping predict attrition.
# 
# Of course, the features mentioned above having the same values may be just due to the rows we have shown here. 
# 
# With that being said, we can get a better picture of which variables only have 1 unique value by looking at the number of unique values for each feature. Also, we will look at each column's type and the number of NA values in each column to see what cleaning might need to be done..
# 

# ## Initial Cleaning
# 
# This section will involve some initial cleaning of the data (mainly removing any unecessary features or filling na values if appropriate).
# 
# ___
# **A Look at Unique Values and NAs**

# In[ ]:


# Display Number of Uniques, If Column has NA, number of NAs, and data types for each column
pd.DataFrame(data = {'Data Type': data.dtypes,
                     'Number of Unique Values': data.nunique().sort_values(),
                     'Contains NAs': data.isnull().any(),
                     'Number of NAs': data.isnull().sum()}).sort_values('Number of Unique Values')


# 
# The data is pretty clean seeing that there are no NA values. Also, we see Over18, StandardHours, and EmployeeCount only have 1 value. Looking back at the first (tabular) look at a few random rows of data, we see Over18 is 'Y', StandardHours is 80, and EmployeeCount is '1'. This means that all employees in this data are over 18, work standard hours of 80, and I assume employee count of 1 corresponds to one row counting as one employee. Employee Count might be used if grouping employees by certain features to count the number of employees in each group.
# 
# EmployeeNumber has 1470 values for 1470 employees, so each employee has an unique employee number (which is somewhat expected). Since Over18, StandardHours, EmployeeCount, and EmployeeNumber either have only 1 value or a unique value for each sample/employee, these variables will not be used in helping predict attrition.

# In[ ]:


cols_to_drop_for_modeling = ['Over18','StandardHours','EmployeeCount','EmployeeNumber']
data.drop(cols_to_drop_for_modeling, axis = 1,inplace=True)


# ## Cateogrical Variable Counts
# 
# Now we will look at the cateogorical variables and see their counts and how the employee counts are distributed
# 

# In[ ]:


cat_cols = list(data.select_dtypes(exclude=np.number).columns)
#cat_cols.remove('Attrition') 

fig_rows = 2
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(cat_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(cat_cols):
    trace = go.Bar(name=col,
                x= data[col].value_counts().index.values,
                y= data[col].value_counts().values,
                marker=dict(line=dict(color='black',width=1)))
    fig.append_trace(trace, curr_row, curr_col)
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1

fig['layout'].update(title =  'Count of Categorical Variables in Dataset', width = 900, height = 600,
                    showlegend=False)

py.iplot(fig)


# 

# Here, we see that none of the other categorical variables shown here is as imbalanced as Attrition, although business travel comes close. 
# 
# Also, looking at department, education field, and jobrole, we can also see these employees look to represent some type of business in the medial research industry.
# 
# Of course, one of the most important ones for this notebook is Attrition, so let us take a look at that one with a larger plot for easier interpretation.

# In[ ]:


attritionBarPlot = go.Bar(
            x= data["Attrition"].value_counts().index.values,
            y= data["Attrition"].value_counts().values,
            marker=dict( color=['Orange', 'steelblue'],line=dict(color='black',width=1)))
layout = dict(title =  'Count of Attrition in Dataset', width = 800, height = 400)
fig = dict(data = [attritionBarPlot], layout=layout)
py.iplot(fig)


# We see there is a significant imbalance between the two classes of our target variable. About 83% have no attrition while the remaining do. There are many techniques to help address a dataset with an imbalanced target variable. These mainly fall in either oversampling (the minority) or undersampling (the majority) type of techniques. 
# 

# ## Numeric Variable Distribuions
# 
# With that data a bit more clean now, we shall look at the distribution of the numeric features in the dataset. It should be noted that some of the variables are read in as integers (numeric), but represent a discrete amount of values. These variables (and the integer meanings if known) are presented below:
# 
# * Education - 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
# * EnvironmentSatisfaction - 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# * JobInvolvement - 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# * JobSatisfaction - 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# * PerformanceRating - 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'
# * RelationshipSatisfaction - 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# * WorkLifeBalance - 1 'Bad' 2 'Good' 3 'Better' 4 'Best'
# * JobLevel
# * StockOptionLevel
# 
# For now, we will treat these as numeric variables. Although these variables have an order to them, their meanings as integers may be flawed.. For example, having a 'Bachelor Degree' (3) does not necessarily mean it is 3 times 'greater' than having 'Below Education' (1) education; however, these features are ordinal (i.e. the categories have an order to them of one being greater than another).
# 
# ___
# **Distributions of Numeric Features**

# In[ ]:


#  Plot areas are called axes
import warnings    # We want to suppress warnings
warnings.filterwarnings("ignore")    # Ignore warnings

fig_rows = 5
fig_cols = 5
fig,ax = plt.subplots(fig_rows,fig_cols, figsize=(16,20)) 
fake_numeric_cols = ['Education','EnvironmentSatisfaction', 'JobInvolvement' , 'JobSatisfaction' , 
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'JobLevel',
                    'StockOptionLevel']
#numericCols = [x for x in list(data.select_dtypes(include=np.number).columns) if x not in fake_numeric_cols]
numericCols = list(data.select_dtypes(include=np.number).columns)
curr_row = 0
curr_col = 0
for i, col in enumerate(numericCols):
    sns.distplot(data[col], ax = ax[(curr_row,curr_col)],rug=True, kde =False) 
    curr_col+=1
    if curr_col >= fig_cols: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 0

plt.show()


# ___
# Here we see histograms of each numeric variable in the dataset along with blue tick marks ontop of the x axis representing where a single observation (employee) lies. This helps visualize the distributions of the numeric variables. For example, we see Age is almost normally distributed with a slight right tail; whereas the rate features (Daily, Monthly, Hourly) look to be close to uniformly distributed.
# 
# It is interesting to note that the Monthly Rate and Monthly Income distributions (and numbers) are very different. I would assume they would beI tried searching for the meaning for these features in other notebooks along with some IBM articles on the dataset, but was unable to get their definitions.
# 
# Similar to some of the information provided in the plots above, we can also take note of some common statistics of the numeric features given below:

# In[ ]:


data[numericCols].describe().T


# Here, we see the count, mean, and standard deviation along with the five number summary. 
# 

# ## Variable/Feature Relationships
# 
# Here we will take a look at how variables related to each other. There are various methods/visualizations for this.
# 
# ### Correlation Matrix
# 
# One method to see how features are related to each other is a correlation matrix given below:
# 
# Note: We convert Attrition to 1 for Yes and 0 for no to be able to see if any variables/features are correlated with the target variable.

# In[ ]:


heatmapCols = numericCols + ['Attrition']
temp = data.copy()
temp['Attrition'] = data['Attrition'].replace('Yes',1).replace('No',0)
heatmapGo = [go.Heatmap(
        z= temp[heatmapCols].astype(float).corr().values, # Generating the Pearson correlation
        x= temp[heatmapCols].columns.values,
        y= temp[heatmapCols].columns.values,
        colorscale='Cividis',
        reversescale = False,
        opacity = 1.0)]

layout = go.Layout(
    title='Pearson Correlation Matrix Numerical Features',
    xaxis = dict(ticks='', tickfont = dict(size = 10)),
    yaxis = dict(ticks='', tickfont = dict(size = 7)),
    width = 900, height = 700)

fig = go.Figure(data=heatmapGo, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# Here we see most attributes are not well correlated with each other (i.e. values close to zero); however, there are a few 'hot' or 'dark' spots in the plot.
# 
# For one, the top right 'hot' area with the yellow/brown colors correspond to the 'year' variables being somewhat correlated with one another. This makes sense. For example, if one has more years with the company, one can reaonsably expect more years in the current role.
# 
# The other interesting hot spot is between Job Level and Monthly Income at just above 0.95! The two rows and two columns corresponding to these features have similar colors. This shows that these two features/variable provide similar information.
# 
# Another interesting thing to note is the lack of correlation between HourlyRate, DailyRate, and MonthlyRate. One would think these variables should correlate well with one another; however, the data seems to indicate otherwise.

# ## Pair Plot
# 
# Another method to view relationship between variables is a pairplot. It is a grid of plots with the same features on the x and y grid axis. It can show distributions along the diagonal (like the histograms in the earlier section). It can also show scatter plots on the off diagonals to show how 2 variables relate. Here, we also color by Attrition to see if any easy/obvious relationships pop up and also only show total working years for the 5 year variables to make plot slightly easier to read.
# 
# Let us first start with the truly numeric vairables:

# In[ ]:


pseudo_numeric_cols = ['Education','EnvironmentSatisfaction', 'JobInvolvement' , 'JobSatisfaction' , 
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'JobLevel',
                    'StockOptionLevel'] 
excludeCols = pseudo_numeric_cols + ['YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','YearsAtCompany']
pairplotCols = [x for x in list(data.select_dtypes(include=np.number).columns) if x not in excludeCols]+ ['Attrition']
sns.pairplot(data[pairplotCols], plot_kws={'scatter_kws': {'alpha': 0.1}},
             kind="reg", diag_kind = "kde"  , hue = 'Attrition' );


# None of the plots show a huge differnce between employees who have stayed (i.e. no attrition) versus employees. I do not like using pair plots for dissecting slight differences as there may be other interactions with other features that this visualization does not show.

# ## Distributions Colored by Attrition
# 
# We can also color the distributions shown earlier by a cateogrical variable. In this case, we will color by attrition since that will be our target variable. Some this was seen in the diagonals in the pair plot shown earlier.
# 
# In addition,  we add a black line showing the percentage of employees that attrition over the numeric variables. This helps give an overall view of how the % of Attrition changes over the range of the numeric features. The y axis on the right side of the plot shows the percentage.

# In[ ]:


numeric_cols = list(data.select_dtypes(include=np.number).columns)

fig_rows = 6
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(numeric_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(numeric_cols):
    trace1 = go.Histogram(name = "No Attrition", 
                          marker=dict( line=dict(color='black',width=1)), #color=['steelblue']),
                          x = list(data[data['Attrition'] == 'No'][col]),
                          opacity=0.5)
    trace2 = go.Histogram(name = 'Yes Attrition', 
                          marker=dict(line=dict(color='black',width=1)),#,color=['Orange']),
                          x=data[data['Attrition'] == 'Yes'][col],
                          opacity=0.5)
    tmp3 = pd.DataFrame(pd.crosstab(data[col],
                    data['Attrition'].replace('Yes',1).replace('No',0)), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    extra_yAxis = 'y' + str(fig_rows * fig_cols + i+1)
    trace3 =  go.Scatter(x=tmp3.index,y=tmp3['Attr%'],
        yaxis = extra_yAxis,name='% Attrition', opacity = .8, 
        marker=dict(color='black',line=dict(color='black',width=0.5)))
    fig.append_trace(trace1, curr_row, curr_col)
    fig.append_trace(trace2, curr_row, curr_col)
    if col not in ['MonthlyRate','DailyRate', 'MonthlyIncome']:
        fig.append_trace(trace3, curr_row, curr_col)
        fig['data'][-1].update(yaxis=extra_yAxis)
        yaxisStr = ''
        if curr_col == fig_cols:
            yaxisStr = '% Attrition'
        fig['layout']['yaxis' + str(fig_rows * fig_cols + i+1)] = dict(range= [0, max(tmp3['Attr%'])+10], 
                         showgrid=True,  overlaying= 'y'+str(i + 1), anchor= 'x'+str(i+1), side= 'right',
                         title= yaxisStr)
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1
fig['layout'].update(title =  'Numerical Distributions colored by Attrition', width = 900, height = 900,
                    barmode = 'overlay',showlegend=False, font=dict(size=10))#,
                    #yaxis2=dict(range= [0, 100], overlaying= 'y', anchor= 'x', 
                    #      side= 'right',zeroline=False,showgrid= False, title= '% Attrition'))

py.iplot(fig)


# In the above plots, there is quite a bit of information that can be taken away. I just want to highlight the following points that mainly focus on attrition rates/percentages (black line):
# 1. Younger Employees (Age) tend to have higher Attrition rates
# - Similarly to Age, employees with lowest level stock options and lowest jobs also tend to have higher attrition rates
# - Also, many of the year features (Total Working Years, Years with Current Manager, Years at Company, etc.) have a general downward trend with regard to attrition rates. In other words, people with less years have higher attrition percentages
# - In addition, many of the survey like features (Job Involvement, Work Life Balance, Job Satisfaction, etc.) tend to have higher attrition percentages the lower an employee scored.
# - There are a few strange observations as the number of employees for a particular range die down. For example, there is one employee with Attrition with 40 total working years (and no other employees in that range), so that results in a '100%' attrition percentage for that range. This can be misleading.
# 
# Let us now take a look at the discrete/categorical features
# 
# **Discrete/Categorical Features Colored by Attrition**

# In[ ]:


cat_cols = list(data.select_dtypes(exclude=np.number).columns)

fig_rows = 2
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(cat_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(cat_cols):
    yaxisStr = ''
    offset_val = -0.3
    if col not in ['Attrition']:
        offset_val = -0.2
    trace1 = go.Bar(name='No Attrition', opacity = .8, width= 0.6,#offset = -0.03,
                x= data[data['Attrition'] == 'No'][col].value_counts().index.values,
                y= data[data['Attrition'] == 'No'][col].value_counts().values,
                marker=dict(color = 'steelblue', line=dict(color='black',width=1)))
    trace2 = go.Bar(name='Yes Attrition', opacity = .8, width= 0.6,offset = offset_val,
                x= data[data['Attrition'] == 'Yes'][col].value_counts().index.values,
                y= data[data['Attrition'] == 'Yes'][col].value_counts().values,
                marker=dict(color = 'orange',line=dict(color='black',width=1)))
    tmp3 = pd.DataFrame(pd.crosstab(data[col],
                    data['Attrition'].replace('Yes',1).replace('No',0)), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    extra_yAxis = 'y' + str(fig_rows * fig_cols + i+1)
    trace3 =  go.Scatter(x=tmp3.index,y=tmp3['Attr%'],mode = 'markers',
        yaxis = extra_yAxis,name='% Attrition', opacity = .8, 
        marker=dict(color='black',size= 10))
    fig.append_trace(trace1, curr_row, curr_col)
    fig.append_trace(trace2, curr_row, curr_col)
    if col not in ['Attrition']:
        fig.append_trace(trace3, curr_row, curr_col)
        fig['data'][-1].update(yaxis=extra_yAxis)
        if curr_col == fig_cols:
            yaxisStr = '% Attrition'
        fig['layout']['yaxis' + str(fig_rows * fig_cols + i+1)] = dict(range= [0, max(tmp3['Attr%'])+10], 
                         showgrid=True,  overlaying= 'y'+str(i + 1), anchor= 'x'+str(i+1), side= 'right',
                         title= yaxisStr)
    
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1

fig['layout'].update(title =  'Count of Categorical Variables in Dataset Colored by Attrition', 
                     width = 900, height = 600,
                     showlegend=False, barmode = 'overlay', font=dict(size=10))

py.iplot(fig)


# Here, the black dots help show the percentage of employees that turned over (left/attrition) for that partuclar value of that category.
# 
# One of the more obvious differences in percentages is Overtime. Of the employees working overtime, about 30% left the company; while only 10% of the employees not working overtime left the company. There are also similar observations for employees with a jobrole of sales representation, employees who travel frequently, and employees who are single.
# 
# 
# # Modeling
# 
# In this section, we will being modeling, but first we will take some notes on preprocessing.
# 
# ## Normalize and Center Numeric Variables
# 
# Some of the models used here will benefit from centering and scaling down the numeric variables. The scaling will be done during CV (in a pipeline) and not the entire dataset (as not to introduce any leakage or undesired effects from the unseen test/validation data)

# In[ ]:


numericCols = list(data.select_dtypes(include=np.number).columns)


# ## One Hot Encode Cateogrical Variables

# In[ ]:



cat_cols = list(data.select_dtypes(exclude=np.number).columns)
if 'Attrition' in cat_cols:
    cat_cols.remove('Attrition')


# In[ ]:


X_cat = pd.get_dummies(data[cat_cols])
X_cat.head()


# ## Modeling
# 
# We shall use roc_auc as our main scoring metric, but will also keep track of accuracy and f1 score (that combine recall/precision).

# In[ ]:


y = data['Attrition'].replace('Yes',1).replace('No',0)


# In[ ]:


X = pd.concat([data[numericCols], X_cat], axis=1, sort=False)
X.head()


# In[ ]:


def get_cv_results(model, X, y):
    pipe = make_pipeline(StandardScaler(),model)
    cv_scores_dict = cross_validate(pipe, X, y, cv=num_folds, 
                                    scoring= ['roc_auc', 'accuracy', 'f1', 'precision','recall'],
                                   return_train_score = False)
    cv_scores_df = pd.DataFrame(cv_scores_dict, 
                 index = ['Fold {}'.format(x) for x in range(1,len(cv_scores_dict['test_accuracy'])+1)])
    return pd.concat([cv_scores_df, pd.DataFrame(cv_scores_df.mean(), columns=['Avg']).T])


# ### Logistic Regression

# In[ ]:


logRegScores = get_cv_results(LogisticRegression(random_state=seed_Num), X, y)
logRegScores


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
naiveBayesScores = get_cv_results(GaussianNB(), X, y)
naiveBayesScores


# ### K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
#knnNeighborComparison = pd.DataFrame()
#for numOfK in range(1,51,2):
#    knnScores = get_cv_results(KNeighborsClassifier(n_neighbors=numOfK), X, y)
#    df = pd.DataFrame(knnScores.loc['Avg']).T
#    df.index = ['K='+str(numOfK) + ' Avg']
#    knnNeighborComparison = pd.concat([knnNeighborComparison,df])
#knnNeighborComparison
## Notes on Iterating through neighbors:
### Seems to plateau around roc_auc of 0.78 and f1 score generally decrease as k increases
### K = 7 seems to be last significant bump in roc_auc with a relatively decent f1
knnScores = get_cv_results(KNeighborsClassifier(n_neighbors=7), X, y)
knnScores


# ### Gradient Boosting Tree based Model (LightGBM)

# In[ ]:


import lightgbm as lgb
lgbmScores = get_cv_results(lgb.LGBMClassifier(random_state=1, n_jobs = -1), X, y)
lgbmScores


# ### SVC

# In[ ]:


from sklearn.svm import SVC
svcScores = get_cv_results(SVC(), X, y)
svcScores


# ### SVC (Linear)

# In[ ]:


from sklearn.svm import SVC
svcLinearScores = get_cv_results(SVC(kernel='linear'), X, y)
svcLinearScores


# ### Neural Net

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlpScores = get_cv_results(MLPClassifier(hidden_layer_sizes = (2,2),
                                         random_state=seed_Num), X, y)
mlpScores


# # Model Comparison and Conclusion
# Here we will compare the average scores of the models.
# 

# In[ ]:


cvAvgResultsCombined = pd.DataFrame()
# Get variables ending with Scores
varNames = [s for s in list(locals().keys()) if s.endswith('Scores')]
local_var_dict = locals()
for varName in varNames:
    df = pd.DataFrame(local_var_dict[varName].loc['Avg']).T
    df.index = [varName+' Avg']
    cvAvgResultsCombined = pd.concat([cvAvgResultsCombined,df])
cvAvgResultsCombined.sort_values(by='test_roc_auc', ascending=False)


# Here we see that the linear models performed the best with regard to receiver operating characteristic area under the curve (linear support vector classifier and logisitic regression). These two also had similar accuracies and f1 scores. A lightly tuned and simple neural net also performed similarily to the linear models. Naive Bayes and K-Nearest Neighbors performed the worst out of the models for this situation.
# 
# # Next Steps
# 
# For next steps, addressing the target class imbalance via undersampleing or oversampling techniques can be explored. For example, SMOTE (Synthetic Minority Over-sampling Technique) can be used to synthetically create more of the minority class (Yes Attrition)  for a more balanced target of Yes/No Attrition.
# 
# Another step to be done is more formal hyperparameter tuning using methods like grid search cv, random search cv, or bayesian hyperparameter optimization. 
# 

# # Sources and References
# 
# For some general ideas and what to visualize (along with what to not visualize), I look at some other notebooks.
# 
# [1] [IBM HR Data Visualization - Devendray](https://www.kaggle.com/devendray/ibm-hr-data-visualization) 
# 
# [2] [Employee Attrition via Ensemble Tree Based Methods - Arthur Tok](https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods)
