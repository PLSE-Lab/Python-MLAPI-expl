#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The dataset is H1B data from Kaggle, spanning from 2011 to 2016. I retained only CERTIFIED and DENIED cases, since these are the two major types and easiest to intepret. This dataset is combined with census data to calculate the fractions of H1B in total population of each state. 
# 
# **[I-Exploratory Data Analysis](#I)** <br>
# EDAs include geographic maps that illustrate the distribution of most common jobs got accepted/denied accross states (normalized to each state population) (*[fig](#fig3)*) <br>
# There are an increase in  total applications submitted from 2011-2016 (*[fig](#fig2)*) <br>
# New Jersey and Washington got the most % (normalized) of H1B acceptance cases, and these applications fall in technology-related jobs (e.g., software developers, computer programmers, computer analysts). 
# 
# **[II-Classfication model](#II)**
# 
# **[Model](#model)**: a simple decision tree classification model to predict the likelihood of getting accepted/rejected for certain H1B case <br>
# Metrics: overall accuracy, precision, recall, AUC <br>
# **[Feature engineering](#feature)** include:
# - Upsampling the minority (denied) class due to the extreme balance of certified vs. denied classes
# - Due to large categorical variables, one-hot-coding is not an optimal solution because of computational cost, so my attempt is to do label encoding. The problem with this approach though is an assumption of ordinal relationship within different level of the categorical features. <br>
# 
# **[Model results](#result)** : . Feature importance suggested that wage is the most crucial feature affecting the acceptance decision. <br>
# AUC is still low (0.59) even when with trying other models such as SVM and logistic regression and there is still room for improvement. Thses include bin-counting treatment for categorical features (to represent each categorical level as the conditional probability of the response variable given that categorical level) and applying a more complex model (e.g., random forest)

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 


# In[4]:


# Import data set
df = pd.read_csv('../input/h-1b-visa/h1b_kaggle.csv')
df.head()


# In[5]:


# Subset the data to get Accepted and Denied cases only since these are the most interpretable
sub_df = df[df['CASE_STATUS'].isin(['CERTIFIED','DENIED'])]
sub_df.info()


# ## I-Exploratory Data Analysis <a class="anchor" id="I"></a>

# In[6]:


#Function to count unique values of each column in the data set
def count_unique(df,col):
    return df[col].value_counts()
print('Case Status','\n', count_unique(sub_df,'CASE_STATUS').head(5))
print('\n')
print('Job Title','\n', count_unique(sub_df,'JOB_TITLE').head(5))


# In[7]:


sub_df.dropna(inplace=True)


# In[8]:


def boxplot(df,col):
    """
    This function plot boxplot of certain feature in each class of target variable
    - df: dataframe
    - col: feature column
    """
    data_to_plot = [df[df['CASE_STATUS']=='CERTIFIED'][col],df[df['CASE_STATUS']=='DENIED'][col]]
    # Create a figure instance
    fig = plt.figure(figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    ax.boxplot(data_to_plot)
    ax.set_xticklabels(['approved','denied'],fontsize=14)
    plt.ylabel(col,fontsize=14)


# In[9]:


boxplot(sub_df,'PREVAILING_WAGE')


# In[11]:


# Create a State column from Worksite 
sub_df['STATE'] = sub_df['WORKSITE'].apply(lambda x:x.split(',')[1])
sub_df.head()


# In[12]:


print(count_unique(sub_df,'JOB_TITLE').head(5))
print('\n')
print(count_unique(sub_df,'SOC_NAME').head(5))


# In[14]:


# Dictionary of state codes. This is used to encode states in our data set
us_state_abbrev = {
    'alabama': 'AL',
    'alaska': 'AK',
    'arizona': 'AZ',
    'arkansas': 'AR',
    'california': 'CA',
    'colorado': 'CO',
    'connecticut': 'CT',
    'delaware': 'DE',
    'district of columbia': 'DC',
    'florida': 'FL',
    'georgia': 'GA',
    'hawaii': 'HI',
    'idaho': 'ID',
    'illinois': 'IL',
    'indiana': 'IN',
    'iowa': 'IA',
    'kansas': 'KS',
    'kentucky': 'KY',
    'louisiana': 'LA',
    'maine': 'ME',
    'maryland': 'MD',
    'massachusetts': 'MA',
    'michigan': 'MI',
    'minnesota': 'MN',
    'mississippi': 'MS',
    'missouri': 'MO',
    'montana': 'MT',
    'nebraska': 'NE',
    'nevada': 'NV',
    'new hampshire': 'NH',
    'new jersey': 'NJ',
    'new mexico': 'NM',
    'new york': 'NY',
    'north carolina': 'NC',
    'north dakota': 'ND',
    'ohio': 'OH',
    'oklahoma': 'OK',
    'oregon': 'OR',
    'pennsylvania': 'PA',
    'rhode island': 'RI',
    'south carolina': 'SC',
    'south dakota': 'SD',
    'tennessee': 'TN',
    'texas': 'TX',
    'utah': 'UT',
    'vermont': 'VT',
    'virginia': 'VA',
    'washington': 'WA',
    'west virginia': 'WV',
    'wisconsin': 'WI',
    'wyoming': 'WY',
}

# Function to encode the states
def code_conversion(df,col):
    Code = []
    for state in df[col]:
        if state.lower().strip(' ') in us_state_abbrev:
            Code.append(us_state_abbrev[state.lower().strip(' ')])
        else:
            Code.append('Null')
    df['CODE'] = Code


# In[15]:


# Encoding states
code_conversion(sub_df,'STATE')
# Subset the data with US states only
US = sub_df[sub_df['CODE'] != 'Null']


# In[16]:


# Aggregate data by state, year, status, and job. Order by and print out top 3 job counts. 
df = US.groupby(['CODE','YEAR','CASE_STATUS','SOC_NAME'])['JOB_TITLE'].count().reset_index()
df.columns = ['CODE','YEAR','CASE_STATUS','SOC_NAME','COUNT']
df_agg = df.groupby(['CODE','YEAR','CASE_STATUS','SOC_NAME']).agg({'COUNT':sum})
g = df_agg['COUNT'].groupby(level=[0,1,2], group_keys=False).nlargest(3).reset_index()
g.head()


# In[17]:


#Create a text column that aggregate job name and count
g['TEXT'] = g['SOC_NAME'].apply(lambda x:x.replace(',','')) + ' ' + g['COUNT'].map(str)
g1 = g.groupby(['CODE','YEAR','CASE_STATUS'])['TEXT'].apply('<br>'.join).reset_index()
g1['TEXT'] = g1['CODE'] + '<br>' + g1['TEXT']
# Create a column of total counts by state, year, and status
g1['TOTAL'] = g.groupby(['CODE','YEAR','CASE_STATUS'])['COUNT'].sum().reset_index()['COUNT']
g1.head()


# In[18]:


# Import census data of estimated state population from 2010 to 2017
census = pd.read_csv('../input/census20102017/nst-est2017-popchg2010_2017.csv')
# Get data from 2011-2016 only
census.drop(['STATE','ESTIMATESBASE2010','POPESTIMATE2010','POPESTIMATE2017'],1,inplace=True)
# Change NAME column to STATE
census = census.rename(columns = {'NAME':'STATE'})
# Encode States
code_conversion(census,'STATE')
# Pivot census table to get population in each year for each state
keys = [c for c in census if c.startswith('POP')]
census = pd.melt(census, id_vars=['STATE','CODE'], value_vars=keys, value_name='POP')
census['YEAR'] = census['variable'].apply(lambda x:int(x[-4:]))
census.head()


# In[19]:


# left join two tables together on 'CODE'
g1_census = pd.merge(g1,census,how='left',on=['CODE','YEAR'])
# Calculate the fraction of H1B in total population for each state in each year
g1_census['PERCENT'] = g1_census['TOTAL']/g1_census['POP'] * 100
g1_census.head()


# In[62]:


# percentages of total submitted cases each year by state
total = g1_census.groupby(['CODE','YEAR'])['PERCENT'].sum().reset_index()


# In[69]:


def yearplot_totalcases(state):
    plt.plot(total[total['CODE']==state]['YEAR'],total[total['CODE']==state]['PERCENT'],label=state)
    plt.ylabel('% Total case submitted (normalized to pop)')
    plt.xlabel('Year')
    plt.legend(loc='best')


# *yearplot_totalcases* <a class="anchor" id="fig2"></a>

# In[71]:


yearplot_totalcases('CA')
yearplot_totalcases('NJ')
yearplot_totalcases('TX')
yearplot_totalcases('WA')


# In[21]:


def geoplot_year(df,year,status,values,title):
    """
    This function plot a geographical map of H1B distribution across states 
    df is the processed data frame g1
    year: 2011 - 2016
    status: 'DENIED' or 'CERTIFIED'
    title: 'denied' or 'accepted'
    """
    if status == 'Null':
        df = df[df['YEAR']==year]
        df['TEXT'] = df['CODE']
    else:
        df = df[(df['YEAR'] == year)&(df['CASE_STATUS']==status)]
    data = dict(type='choropleth',
            colorscale = 'YIOrRd',
            locations = df['CODE'],
            z = df[values],
            locationmode = 'USA-states',
            text = df['TEXT'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"%"}
            ) 
    layout = dict(title = '%d Percentage of H1B cases %s normalized to state population' %(year,title),
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )
    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap)


# *geographical plots* <a class="anchor" id="fig3"></a>

# In[42]:


geoplot_year(g1_census,2016,'CERTIFIED','PERCENT','accepted')
geoplot_year(g1_census,2016,'DENIED','PERCENT','denied')


# In[77]:


# Plot the % of total H1B cases in each state (normalized to state population)
geoplot_year(total,2016,'Null','PERCENT','Submitted')


# ## II-Classfication model  <a class="anchor" id="II"></a>
# Given a set of features, this will predict how likely a case will be accepted. Due to a mix of categorical and continuous variables, a tree-based model is chosen. Label encoder is applied to deal with categorical features. Two classes of CASE STATUS are predicted (CERTIFIED = 0 and DENIED = 1)

# In[24]:


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import itertools
from imblearn.over_sampling import RandomOverSampler


# ### Functions to plot classification metrics <a class="anchor" id="model"></a>

# In[25]:


def plot_confusion_matrix(cm, y_test,model_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Classification report','\n',classification_report(y_test,model_pred))
    print('\n')
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[26]:


def plot_feature_importance(model):
    
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    
    #If the model used is tree-based
    df_f = pd.DataFrame(model.feature_importances_, columns=["importance"])
           
    # Display table and plot of the feature importance 
    df_f["labels"] = features
    df_f.sort_values("importance", inplace=True, ascending=False)
    display(df_f.head(5))
    index = np.arange(df_f.shape[0])

    bar_width = 0.5
    rects = plt.barh(index , df_f["importance"], bar_width, alpha=0.4, color='b', label='Main')
    plt.yticks(index, df_f["labels"],fontsize=30)
    plt.xticks(fontsize=30)


# In[27]:


def plot_ROC(model,y_test,model_pred):
    """
    This function plots the ROC curve and print out AUC
    """
    probs = model.predict_proba(X_test)
    cm = pd.DataFrame(confusion_matrix(y_test, model_pred),
                                columns=["Predicted False", "Predicted True"], 
                                index=["Actual False", "Actual True"])
    display(cm)

    # Calculate the fpr and tpr for all thresholds of the classification
    plt.figure(figsize=(10,6))
    fpr, tpr, threshold = roc_curve(y_test, probs[:,1])
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.xlabel('False Positive Rate',fontsize=14)
    print('Area under the curve is %f' %roc_auc_score(y_test,model_pred))


# ### Data engineering and building predictor <a class="anchor" id="feature"></a>

# In[28]:


# Convert the categorical feature to numeric feature
def label_encoder(df,col):
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes

cols = ['CASE_STATUS','CODE','SOC_NAME','FULL_TIME_POSITION','EMPLOYER_NAME','JOB_TITLE']
for col in cols:
    label_encoder(sub_df,col)


# In[29]:


sub_df.dropna(inplace=True)


# In[30]:


# Prepare to build model, start with train, test splitting
X = sub_df[['CODE','SOC_NAME','FULL_TIME_POSITION','EMPLOYER_NAME','JOB_TITLE','PREVAILING_WAGE']]
y = sub_df['CASE_STATUS']
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[31]:


# Oversampling the minor class (status = 1, denied)
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)


# In[76]:


# Examine the new y_train to see if there is class balance 
pd.Series(y_resampled).value_counts()


# In[32]:


# Fit a simple decision tree classifier
dtree = DecisionTreeClassifier()
dtree.fit(X_resampled,y_resampled)
dtree_pred = dtree.predict(X_test)


# ### Model results  <a class="anchor" id="result"></a>

# In[33]:


plot_feature_importance(dtree)


# In[34]:


plot_ROC(dtree,y_test,dtree_pred)


# In[35]:


cm = confusion_matrix(y_test,dtree_pred)
target_names = ['0','1']
plot_confusion_matrix(cm, y_test,dtree_pred,classes=target_names,title='Confusion matrix')


# In[ ]:




