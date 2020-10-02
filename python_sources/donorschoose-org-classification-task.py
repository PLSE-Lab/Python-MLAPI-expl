#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
from numpy import array
from pandas import DataFrame,Series
import matplotlib
import matplotlib.pyplot as plt # for plotting
import matplotlib.patches as patches
from matplotlib import cm
import seaborn as sns # for making plots with seaborn
color = sns.color_palette() # init color object 
import plotly.offline as py # create embed interactive plots
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go 
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
from scipy import interp
import squarify
import re
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
# Print all rows and columns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
from sklearn import preprocessing
from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestClassifier())
# plot arrows


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from subprocess import check_output
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


def generate_data_audit(data,file_name):
    """
    This function process the DataFrame and create a csv file with passed on file_name.
    """
    d=data.dtypes[data.dtypes!=('object')].index.values
    data[d]=data[d].astype('float64')
    mean=DataFrame({'mean':data[d].mean()})
    std_dev=DataFrame({'std_dev':data[d].std()})
    missing= DataFrame({'missing':data[d].isnull().sum()})
    obs=DataFrame({'obs':np.repeat(data[d].shape[0],len(d))},index=d)
    missing_perc=DataFrame({'missing_perc':data[d].isnull().sum()/data[d].shape[0]})
    minimum=DataFrame({'min':data[d].min()})
    maximum=DataFrame({'max':data[d].max()})
    unique=DataFrame({'unique':data[d].apply(lambda x:len(x.unique()),axis=0)})
    q5=DataFrame({'q5':data[d].apply(lambda x:x.dropna().quantile(0.05))})
    q10=DataFrame({'q10':data[d].apply(lambda x:x.dropna().quantile(0.10))})
    q25=DataFrame({'q25':data[d].apply(lambda x:x.dropna().quantile(0.25))})
    q50=DataFrame({'q50':data[d].apply(lambda x:x.dropna().quantile(0.50))})
    q75=DataFrame({'q75':data[d].apply(lambda x:x.dropna().quantile(0.75))})
    q85=DataFrame({'q85':data[d].apply(lambda x:x.dropna().quantile(0.85))})
    q95=DataFrame({'q95':data[d].apply(lambda x:x.dropna().quantile(0.95))})
    q99=DataFrame({'q99':data[d].apply(lambda x:x.dropna().quantile(0.99))})
    DQ=pd.concat([mean,std_dev,obs,missing,missing_perc,minimum,maximum,unique,q5,q10,q25,q50,q75,q85,q95,q99],axis=1)

    c=data.dtypes[data.dtypes=='object'].index.values
    Mean=DataFrame({'mean':np.repeat('Not Applicable',len(c))},index=c)
    Std_Dev=DataFrame({'std_dev':np.repeat('Not Applicable',len(c))},index=c)
    Missing=DataFrame({'missing':data[c].isnull().sum()})
    Obs=DataFrame({'obs':np.repeat(data[d].shape[0],len(c))},index=c)
    Missing_perc=DataFrame({'missing_perc':data[c].isnull().sum()/data[c].shape[0]})
    Minimum=DataFrame({'min':np.repeat('Not Applicable',len(c))},index=c)
    Maximum=DataFrame({'max':np.repeat('Not Applicable',len(c))},index=c)
    Unique=DataFrame({'unique':data[c].apply(lambda x:len(x.unique()),axis=0)})
    Q5=DataFrame({'q5':np.repeat('Not Applicable',len(c))},index=c)
    Q10=DataFrame({'q10':np.repeat('Not Applicable',len(c))},index=c)
    Q25=DataFrame({'q25':np.repeat('Not Applicable',len(c))},index=c)
    Q50=DataFrame({'q50':np.repeat('Not Applicable',len(c))},index=c)
    Q75=DataFrame({'q75':np.repeat('Not Applicable',len(c))},index=c)
    Q85=DataFrame({'q85':np.repeat('Not Applicable',len(c))},index=c)
    Q95=DataFrame({'q95':np.repeat('Not Applicable',len(c))},index=c)
    Q99=DataFrame({'q99':np.repeat('Not Applicable',len(c))},index=c)
    dq=pd.concat([Mean,Std_Dev,Obs,Missing,Missing_perc,Minimum,Maximum,Unique,Q5,Q10,Q25,Q50,Q75,Q85,Q95,Q99],axis=1)

    DQ=pd.concat([DQ,dq])
    DQ.to_csv(file_name)
    


# In[ ]:


from nltk.corpus import stopwords #To check the list of stopwords 

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
    text: a string
    return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()


# In[ ]:


# Read the files in data frames
df_es = pd.read_csv("/kaggle/input/outcome-value/essays.csv/essays.csv")
df_out = pd.read_csv("/kaggle/input/outcome-value/outcomes.csv/outcomes.csv")
df_proj = pd.read_csv("/kaggle/input/outcome-value/projects.csv/projects.csv")


# In[ ]:


# Join the data frames based on the project id, leaving out rows which dont have any outcome/classified yet
df = (df_proj.merge(df_es, left_index=True, right_index=True,
                 how='inner', suffixes=('', '_y'))).merge(df_out, left_index=True, right_index=True,
                 how='inner', suffixes=('', '_y'))  
df.drop(list(df.filter(regex='_y$')), axis=1, inplace=True)


# """print(df_out.shape)
# print(df_proj.shape)
# print(df_es.shape)
# print(df_out.head())
# print(df_proj.head())
# print(df_es.head())"""

# In[ ]:


print(df_out.shape)
print(df_proj.shape)
print(df_es.shape)
print(df.shape)


# In[ ]:


# To free up some memory use garbage collector and set the initial data frames as null
import gc
df_es=pd.DataFrame()
df_out=pd.DataFrame()
df_proj=pd.DataFrame()
del [[df_es,df_out,df_proj]]
gc.collect()


# # A look into the data tell us that lots of column has value 't','f' and nan as unique values. <br>
# We need to remap t as 1 and f as 0.

# In[ ]:


# lets create a Directory and see what are the columns which have  only 3 unique values in them
unique_dict = {}
for col in df.columns:
    if len(df[col].unique())<4:
        unique_dict[col] = df[col].unique()


# To Simplify computation lets mark the nan as 0 for the below columns - 
# at_least_1_teacher_referred_donor<br>
# at_least_1_green_donation<br>
# three_or_more_non_teacher_referred_donors<br>
# one_non_teacher_referred_donor_giving_100_plus<br>
# donation_from_thoughtful_donor<br>

# In[ ]:


#cretaed the map assuming NAN as 0
t_f_map ={'t': 1, 'f': 0,np.nan:0}


# In[ ]:


# for all columns apply the map. This will transform categorical to numeric variables. we dont require on hot Encoding here
for col in unique_dict.keys():
    df[col]= df[col].map(t_f_map)


# In[ ]:


#create a Data Audit Report 
generate_data_audit(df,'data_audit.csv')


# Quick check<br>
# df['at_least_1_green_donation'] <br>
# df.info()<br>
# df.describe()
# 

# In[ ]:


# get the missing data in number/Percentages
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(50)


# missing_data.head(50)

# In[ ]:


# with almost 30% blank value in secondary_focus_subject and secondary_focus_area there is no point in keeping this field in prediction model as there is no way to gather information for this.
df = df.drop(['secondary_focus_area','secondary_focus_subject'],axis =1)


# df.to_csv('output.csv')

# In[ ]:


# check the distribution of target class
temp = df['is_exciting'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Project proposal is approved or not')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # clearly there is a class imbalance in the data under analysis

# In[ ]:


# check the distribution of proposal per state
temp = df["school_state"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of School states in % ",
    xaxis=dict(
        title='State Name',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')


# In[ ]:


# Pictorial Display of state importance 
temp = pd.DataFrame(df["school_state"].value_counts()).reset_index()
temp.columns = ['state_code', 'num_proposals']

data = [dict(
        type='choropleth',
        locations= temp['state_code'],
        locationmode='USA-states',
        z=temp['num_proposals'].astype(float),
        text=temp['state_code'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Number of project proposals'),
)]
layout = dict(title = 'Project Proposals by US States',geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# # So its clear CA has the higest number of submission and next based is NY. <br>
# This seems to be a important variable to consider. Needs to check if we can bin it some categories.

# In[ ]:


# Check the proposal Disrtibution per Grade
temp = df["grade_level"].value_counts()
print("Total number of project grade categories : ", len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of project_grade_category (school grade levels) in %",
    xaxis=dict(
        title='school grade levels',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in % ',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')


# Need to get dummies for this columns. Out of 4 school grade levels, Project proposals submission in school grade levels is higher for **Grades Prek-2** which is approximately **41 %** followed by **Grades 3-5** which has approx. **34 %**.

# # Need to check impact of primary_focus_subject and primary_focus_area
# 
# 

# In[ ]:


print (df['primary_focus_area'].unique())
print(df['primary_focus_subject'].unique())


# # Create visualization of For  primary_focus_area and primary_focus_subject

# In[ ]:


temp = df["primary_focus_area"].value_counts()
print("Total number of project based on focus area : ", len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of primary_focus_area (school grade levels) in %",
    xaxis=dict(
        title='Primary focus area',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in % ',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='focusArea')


# In[ ]:


temp = df["primary_focus_subject"].value_counts()
print("Total number of project based on focus subject : ", len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of primary_focus_Subject (school grade levels) in %",
    xaxis=dict(
        title='Primary focus Subject',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in % ',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='focusSubject')


# In[ ]:


total_cnt = df["primary_focus_area"].value_counts()
total_exiting = df["primary_focus_area"][df['is_exciting']==1].value_counts()
impact  = pd.concat([total_cnt, total_exiting], axis=1, keys=['Total', 'existing'])
impact['percentage_sucess'] = (impact['existing']/impact['Total'])*100
#impact.head(25)


# In[ ]:


# droping the primary_focus_subject parameter as this parameter is not providing any more insight than simply using primary_focus_area
# more over the success percentage is simmilar accross Category.
df = df.drop(['primary_focus_subject'], axis =1)


# df['is_exciting'][df['teacher_prefix'].isnull()]

# In[ ]:


# Sucess and rejection based on state
temp = df["school_state"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(df["is_exciting"][df["school_state"]==val] == 1))
    temp_y0.append(np.sum(df["is_exciting"][df["school_state"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular School states in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Again The importance of CA/NY are far more than other states from FL onwards.

# In[ ]:


# Project Proposals Mean Acceptance Rate by US States

temp = pd.DataFrame(df.groupby("school_state")["is_exciting"].apply(np.mean)).reset_index()
temp.columns = ['state_code', 'num_proposals']

data = [dict(
        type='choropleth',
        locations= temp['state_code'],
        locationmode='USA-states',
        z=temp['num_proposals'].astype(float),
        text=temp['state_code'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Number of project proposals'),
)]
layout = dict(title = 'Project Proposals Mean Acceptance Rate by US States',geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# In[ ]:


df["school_state"].value_counts()


# In[ ]:


# Check impact of Teacher's Prefix in outcome
temp = df["teacher_prefix"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(df["is_exciting"][df["teacher_prefix"]==val] == 1))
    temp_y0.append(np.sum(df["is_exciting"][df["teacher_prefix"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular Teacher prefixes in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# There is a clear imbalance of gender vs submisition

# In[ ]:


# numeric distribution of prefix
df['teacher_prefix'].value_counts()


# print(df['school_metro'].value_counts())
# print('Number of unique county - ',len(df['school_county'].unique()))
# print('Number of unique school_metro - ',len(df['school_metro'].unique()))
# print('Number of unique school_city - ',len(df['school_city'].unique()))
# 

# In[ ]:


# Dropping the location attributes and id as we will be using state as location parameter. 
df = df.drop(['school_metro','school_zip','school_city','school_longitude','school_ncesid','school_latitude'], axis =1)

                                            


# In[ ]:


#cretaed the map to get Female as 1 and Male as 0. In doctor we might have females, but considering the population we are assuming them to 0
gender_map ={'Mrs.': 1, 
          'Ms.':1,
          'Mr.': 0,
          'Dr.':0,
          'Mr. & Mrs.':0}


# In[ ]:


# update the features with the map
df['teacher_prefix'] = df['teacher_prefix'].map(gender_map)


# In[ ]:


df["date_posted"] = pd.to_datetime(df["date_posted"])
df["month_created"] = df["date_posted"].dt.month_name()
df["year"] = df["date_posted"].dt.year
df["weekday_created"] = df["date_posted"].dt.weekday_name
df = df.drop('date_posted',axis =1)


# In[ ]:


temp = df["month_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(df["is_exciting"][df["month_created"]==val] == 1))
    temp_y0.append(np.sum(df["is_exciting"][df["month_created"]==val] == 0))
    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Month Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


temp = df["weekday_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(df["is_exciting"][df["weekday_created"]==val] == 1))
    temp_y0.append(np.sum(df["is_exciting"][df["weekday_created"]==val] == 0))
 
temp.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission weekday Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


#Teacher_prefix and is_exciting Intervals Correlation
cols = ['teacher_prefix', 'is_exciting']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(df[cols[0]], df[cols[1]]).style.background_gradient(cmap = cm)


# In[ ]:


#Correlation Matrix
corr = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation between different features')
pd.DataFrame(corr).to_csv('corr.csv')
corr


# # There is not much corelation except total_price_excluding_optional_support and total_price_including_optional_support.
# Created a normalized price columns from . This is higly correlated feature
# 

# In[ ]:


df['price'] = df['total_price_excluding_optional_support']+ df['total_price_including_optional_support']


# In[ ]:


df = df.drop(['total_price_excluding_optional_support','total_price_including_optional_support'], axis =1)


# In[ ]:


second_analysis_false = {}
second_analysis_true = {}
for col in df.columns :
    if len(df[col].unique())<30:
        second_analysis_false[col] = df[col][df['is_exciting']==0].value_counts().to_frame()
        second_analysis_true[col] = df[col][df['is_exciting']==1].value_counts().to_frame()
    


# In[ ]:


with open('second_analysis_fasle.csv', 'w') as f:
    for key in second_analysis_false.keys():
        f.write("%s,%s\n"%(key,second_analysis_false[key]))


# In[ ]:


with open('second_analysis_true.csv', 'w') as f:
    for key in second_analysis_true.keys():
        f.write("%s,%s\n"%(key,second_analysis_true[key]))


# ![![image.png](attachment:image.png)](http://)
# Clearly shows the importance of the parameters fully_funded
# at_least_1_green_donation
# great_chat
# three_or_more_non_teacher_referred_donors
# one_non_teacher_referred_donor_giving_100_plus

# # Text Processing

# In[ ]:


# Word imporatcance cloud
temp_data = df.dropna(subset=['short_description'])
# converting into lowercase
temp_data['short_description'] = temp_data['short_description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['short_description'] = temp_data['short_description'].map(text_prepare)

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['short_description'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Word Cloud of short_description", fontsize=35)
plt.axis("off")
plt.show() 


# In[ ]:


# quick view on the title parameter's effect on the proposal
temp = df["title"].value_counts().head(25)
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(df["is_exciting"][df["title"]==val] == 1))
    temp_y0.append(np.sum(df["is_exciting"][df["title"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular project titles in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # Text features extraction

# In[ ]:


df['easay_len'] = df['essay'].apply(lambda x: len(str(x))) # Essay length
df['need_statement_len'] = df['need_statement'].apply(lambda x: len(str(x))) # Need Statement length
df ['short_description_len'] = df['short_description'].apply(lambda x: len(str(x))) # Short description length
df['title_len']=df['title'].apply(lambda x: len(str(x))) # title length


# In[ ]:


temp = pd.DataFrame()
temp['text'] = df.apply(lambda row: ' '.join([str(row['essay']), 
                                            str(row['need_statement']),
                                            str(row['short_description']),
                                            str(row['title'])
                                            ]), axis=1)


# In[ ]:


df = df.drop(['essay','need_statement','short_description','title'],axis =1)


# In[ ]:


df['char_count'] = temp['text'].apply(len)
df['word_count'] = temp['text'].apply(lambda x: len(x.split()))


# In[ ]:


df['word_density'] = df['char_count'] / (df['word_count']+1)


# In[ ]:


df['word_count'] = temp['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))


# In[ ]:


df['upper_case_word_count'] = temp['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# In[ ]:


df['stopword_count'] = temp['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))


# In[ ]:


df['punctuation_count'] = temp['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 


# In[ ]:


generate_data_audit(df,'data_audit_before_Cat2Numeric.csv')


# 
# 
# 
# 
# 
# 
# 
# 

# (df['is_exciting'][df['teacher_prefix'].isnull()]).value_counts()

# (df['is_exciting'][df['grade_level'].isnull()]).value_counts()

# df['grade_level'].value_counts()

# df.groupby(['resource_type','is_exciting']).size()

# df.groupby(['resource_type','is_exciting']).size()
# resource_type  is_exciting
# Books          0.0            124706<br>
#                1.0              7793<br>
# Other          0.0             62217<br>
#                1.0              3971<br>
# Supplies       0.0            198651<br>
#                1.0             12475<br>
# Technology     0.0            193220<br>
#                1.0             12195<br>
# Trips          0.0              3189<br>
#                1.0               223<br>
# Visitors       0.0               595<br>
#                1.0                46<br>
# dtype: int64

# df.groupby(['fulfillment_labor_materials','is_exciting']).size()<br>
# 
# fulfillment_labor_materials  is_exciting<br>
# 9.0                          0.0             60308<br>
#                              1.0              3787<br>
# 17.0                         0.0             87431<br>
#                              1.0              5505<br>
# 30.0                         0.0            226703<br>
#                              1.0             14399<br>
# 35.0                         0.0            208129<br>
#                              1.0             13016<br>
# 
# 
# 30.0    241102<br>
# 35.0    221145<br>
# 17.0     92936<br>
# 9.0      64095<br>

# ******Column With missing values** **- **
# 1. teacher_prefix - 4 Will be replaced with 0 as (df['is_exciting'][df['teacher_prefix'].isnull()]).value_counts() has only 0<br>
# 2. fulfillment_labor_materials - 48, Replacing 48 with 0 and create Bin '0'<17.1<'1' <br>
# . students_reached - 146 replacing with mean <br>
# 3. great_messages_proportion - 179839 Dropping  because the missing percentage is very high and number of unique values are high. <br> 
# 4. teacher_referred_count - 94398 Dropping  because the missing percentage is very high and number of unique values are high. <br>
# 5. non_teacher_referred_count - 94398  Dropping  because the missing percentage is very high and number of unique values are high.<br>
# 6. school_district - 548 Dropping as Anyway considering state as location parameter<br>
# 7. primary_focus_area - 39 , Marking as Literacy & Language as this is the higest category by count by a big percentage<br>
# 8. resource_type - 45 , Marking this as Other category<br>
# 9. grade_level - 9 , We have already a class imbalance in our data set. All the observations has is_exciting as 0. droping this records <br>
# 10. dropping school_county as using state as paramter
# 
# 

# In[ ]:


print(df.shape)
df['teacher_prefix'].fillna(0, inplace = True)
df['fulfillment_labor_materials'].fillna(0, inplace = True)
df['fulfillment_labor_materials'] = df['fulfillment_labor_materials'].apply(lambda x:1 if x>17 else 0)
df['primary_focus_area'].fillna('Literacy & Language', inplace = True)
df['resource_type'].fillna('Other', inplace = True)
df['students_reached'].fillna(df['students_reached']. mean(), inplace = True)
df['grade_level'].dropna( axis =0,inplace = True)
column_to_drop = ['non_teacher_referred_count','teacher_referred_count','great_messages_proportion','school_county','school_district','teacher_acctid','schoolid']
df = df.drop(column_to_drop, axis =1)
print(df.shape)


# # Normalize the data to get  better predictions

# In[ ]:


#columns to apply normailzation
apply_normalization = ['students_reached','price','easay_len','need_statement_len',
'short_description_len','title_len','char_count','word_count',
'word_density','upper_case_word_count','stopword_count','punctuation_count'
]

for i in apply_normalization:
    df[i] = (df[i]-df[i].mean())/(df[i].max() -df[i].min())
    


# In[ ]:


# Create a CSV to revalidate the data under process after transformations
generate_data_audit(df,'data_audit_before_Cat2Numeric_after Transform.csv')


# kl = df.groupby(['school_state','is_exciting']).size()
# kl.to_csv('State_analysis.csv')
# 
# df_s['Success_analysis']= (df_s['Success per'] - df_s['Success per'].mean())*100/(df_s['Success per'].max()-df_s['Success per'].min()) <br>
# 
# bins = [-100, -20, 0, 20, 100]<br>
# labels = ['A','B','C','D']<br>
# 
# df_s['binned'] = pd.cut(df_s['Success_analysis'], bins=bins, labels=labels)<br>

# # Put States in Bin based on success rates

# In[ ]:


state_map_bin = {'WY' : 'D','MT' : 'D','RI' : 'D','WV' : 'D','DC' : 'D',
'VT' : 'D','OK' : 'D','MN' : 'D','NM' : 'C','LA' : 'C','SD' : 'C',
'WA' : 'C','VA' : 'C','KY' : 'C','GA' : 'C','UT' : 'C','MA' : 'C',
'TX' : 'C','HI' : 'C','IA' : 'C','SC' : 'C','KS' : 'C','FL' : 'C',
'OH' : 'C','ME' : 'C','NY' : 'B','MD' : 'B','NH' : 'B','IL' : 'B',
'IN' : 'B','CA' : 'B','MS' : 'B','PA' : 'B','CO' : 'B','MI' : 'B',
'AR' : 'B','NC' : 'B','OR' : 'B','MO' : 'B','AZ' : 'B','WI' : 'B',
'NV' : 'B','DE' : 'B','NE' : 'A','TN' : 'A','AL' : 'A','CT' : 'A',
'NJ' : 'A','ND' : 'A','AK' : 'A','ID' : 'A'
}

df['school_state'] = df['school_state'].map(state_map_bin)


# # Transform the catgorical to numerical using one hot encoding

# In[ ]:



cols = ['school_state', 'grade_level','primary_focus_area', 'resource_type','poverty_level','month_created','weekday_created']
df_dummies = pd.get_dummies(df, columns =cols, drop_first = True)
df_dummies.shape


# # for analysis write Sample data to a CSV
# (df_dummies.head(50)).to_csv('final_data.csv')

# In[ ]:


(df_dummies['year'][df_dummies['year']!= 2014]).value_counts()


# # Create Test Train split based on year 2014

# In[ ]:


X_test = df_dummies[df_dummies['year']== 2014]
X_train = df_dummies[df_dummies['year']!= 2014]
y_train = (X_train['is_exciting'])
y_train = y_train.astype(int)
y_test = X_test['is_exciting'].astype(int)
project_id = X_test['projectid'] # to get the prediction report
cl_to_drop = ['is_exciting','year','projectid']
X_test.drop(cl_to_drop,axis =1, inplace = True)
X_train.drop(cl_to_drop,axis =1, inplace = True)


# In[ ]:


print('  \n '.join((' X- Test',str(X_test.shape[0]),
                    'y_test',str(y_test.shape[0]), 
                    'X_train',str(X_train.shape[0]),
                    'y_train',str(y_train.shape[0])
                                            )))


# In[ ]:


# Create a Random Forest Model
random_forest = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)


# # Compared Different Models
# 
# models = { 'kNN': KNeighborsClassifier(n_neighbors=3),<br>
#           'NB'  : GaussianNB(),<br>
#           'regr' :random_forest,<br>
#           'logistic': LogisticRegression(random_state=0,max_iter=500)<br>
#                    }

# In[ ]:


# reduced the model to 2
models = { 'regr' :random_forest,
          'logistic': LogisticRegression(random_state=0,max_iter=500)
          }


# In[ ]:


"""
Function to create Precision recal Graph 
"""

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
    plt.figure(figsize=(15, 8))
    plt.show()


# In[ ]:


# Data Frame to Capture Model evaluation parameter
model_auc = pd.DataFrame(columns= ['Model','Fold','AUC'])


# In[ ]:


"""
This method create a comparison analysis of different models
"""

def create_model_compare(model):
    model_name = type(model).__name__
    print ("###############################################")
    print("Create Model for : ", model_name )
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111,aspect = 'equal')
    ax1.add_patch(
        patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
        )
    ax1.add_patch(
        patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
        )

    tprs = []
    aucs = []
    scores = []
    results = pd.DataFrame(columns=['training_score', 'test_score'])
    mean_fpr = np.linspace(0,1,100)
    i = 1
    cv = StratifiedKFold(n_splits=5, random_state=100,shuffle=True)
   
    for (train, test), i in zip(cv.split(X_train, y_train), range(5)):
        prediction = model.fit(X_train.iloc[train],y_train.iloc[train]).predict_proba(X_train.iloc[test])
        fpr, tpr, t = metrics.roc_curve(y_train.iloc[test], prediction[:, 1])
        #plot_precision_and_recall(fpr, tpr, t)
        tprs.append(interp(mean_fpr, fpr, tpr))

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()
    print ("###############################################")


# In[ ]:


# To store the models
final_model ={}


# In[ ]:


for name , model in models.items():
    fit_model = model.fit(X_train, y_train)
    pred = fit_model.predict(X_test)
    skfold = StratifiedKFold(n_splits=5, random_state=100,shuffle=True)
    results_skfold = model_selection.cross_val_score(model, X_train, y_train, cv=skfold)
    for count,ele in enumerate(results_skfold,1):
        # Initialise data to lists. 
        data = [{'Model': name, 'Fold': count, 'AUC':ele}] 
        temp = pd.DataFrame(data)
        model_auc = model_auc.append(temp)
    
    score = metrics.accuracy_score(y_test,pred)
    print("{:3s}: {:0.2f}".format(name,score))
    print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
    create_model_compare(model)## Run this only at training stage - should comment this out if dont require detail compared.
    final_model[name] = fit_model
    


# In[ ]:


# write the matrix to a CSV file for evaluation
model_auc.to_csv('MODEL_FOLD_AUC.csv')


# In[ ]:


# Choosing the logstic regression model as this is simple and have better prediction

final_ml_model = final_model.get('logistic')


# In[ ]:


# get the probablity prediction from the model for class 1
y_pred= final_ml_model.predict_proba(X_test)[:, 1]
y_pred= pd.Series(y_pred)
print (y_pred.shape)


# In[ ]:


# create a Data frmae to hold the prediction and project id. The Labvel is the original label from the data
final_prediction_csv= pd.DataFrame(columns = ['is_exciting','projectid','Label'] )
final_prediction_csv['is_exciting'] = np.around(y_pred, decimals=6)
final_prediction_csv['projectid'] = project_id
final_prediction_csv['Label'] = y_test


# In[ ]:


# Create a CSV of predictions.
final_prediction_csv.to_csv('final_prediction.csv')

