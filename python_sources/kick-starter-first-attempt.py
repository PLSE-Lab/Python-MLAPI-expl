#!/usr/bin/env python
# coding: utf-8

# # Contents
# 
# - <a href='#Importthelibraries'>Import the libraries</a>
# - <a href='#DataCollection'>Data Collection</a>
# - <a href='#DataDescription'>Data Description</a>
# - <a href='#DataPreprocessing'>4. Data Preprocessing</a>
#    - <a href='#State'>State</a>
#    - <a href='#Name'>Name</a>
#    - <a href='#CategoryandMainCategory'>Category and Main Category</a>
#    - <a href='#Country'>Country</a>
#    - <a href='#Currency'>Currency</a>
#    - <a href='#DeadlineandLaunched'>Deadline and Launched</a>
#    - <a href='#Goal'>Goal</a>
#    - <a href='#DataDeletion'>Data Deletion</a>
#    - <a href='#DataConversion'>Data Conversion</a>
# - <a href='#DataModelling'>5. Data Modelling</a>
#    - <a href='#TrainingsetandTestset'>Training set and Test set</a>
#    - <a href='#DecisionTreeClassifier'>Decision Tree Classifier</a>
#    - <a href='#RandomForestClassifier'>Random Forest Classifier</a>
#    - <a href='#LogisticRegression'>Logistic Regression</a>
# - <a href='#Conclusion'>6. Conclusion</a>
# 

# #  <a id='Importthelibraries'>Import the libraries</a>

# ## Data Analysis libraries

# In[190]:


import numpy as np
import pandas as pd


# ## Data Visualization libraries

# In[191]:


import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Model Building libraries

# In[192]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# # <a id='DataCollection'>Data Collection</a>

# In[193]:


data=pd.read_csv('../input/ks-projects-201801.csv')


# # <a id='DataDescription'>Data Description</a>

# In[194]:


data.shape


# In[195]:


data.head()


# In[196]:


data.columns.values


# #### A Function to help along the way

# In[197]:


def cat_feat(col):
    x=pd.DataFrame(data={'count':data[col].value_counts(),
                         'occurence rate':data[col].value_counts()*100/data.shape[0]},
                  index=data[col].unique())
    y=data[[col,'state']].groupby(col).mean()
    y['state']=y['state']*100
    y.rename(columns={'state':'success rate'},inplace=True)
    return pd.concat([x,y],axis=1)


# In[198]:


col_details=pd.DataFrame(columns=['Null','Unique','Action'],index=data.columns.values)
for col in data:
    col_details.loc[col]['Null']=data[col].isnull().sum()
    col_details.loc[col]['Unique']=data[col].nunique()
col_details


# # <a id='DataPreprocessing'>Data Preprocessing</a>

# >## <a id='State'>Reducing values for 'State'</a>

# #### Different values for the state attribute and their freuency

# In[199]:


data['state'].value_counts()


# There are 6 states for the final outcome of any project:
# * Failed
# * Successful
# * Canceled
# * Undefined
# * Live
# * Suspended
# 
# We are going to reduce these to only 2 states:
# * Successful or 1
# * Unsuccessful or 0

# #### Some inconsistencies.....

# Projects are successful but real goal < pledged goal

# In[200]:


data[(data['state']=='successful')&(data['usd_pledged_real']<data['usd_goal_real'])]


# Projects are unsuccessful but real goal>= goal

# In[201]:


data[(data['state']=='failed')&(data['usd_pledged_real']>=data['usd_goal_real'])]


# A project is successful if:
#     1. It is already marked successful
#     2. The pledged money is greater than or equal to the goal.

# In[202]:


data.at[data[(data['state']=='successful')|(data['usd_pledged_real']>=data['usd_goal_real'])].index,'state']=1


# A project is unsuccessful if the pledged money is less than the goal and the current state is not 'live'

# In[203]:


data.at[data[(data['state']!='live')&(data['usd_pledged_real']<data['usd_goal_real'])].index,'state']=0


# We won't be considering live projects for our model because the final state is not known. But it can be used as a test set to predict their outcomes.

# In[204]:


test=data[data['state']=='live'].copy()
test.drop('state',axis=1,inplace=True)
data.drop(data[data['state']=='live'].index,inplace=True,axis=0)


# In[205]:


data['state']=pd.to_numeric(data['state'])


# >##  <a id='Name'>Working with the name attribute</a>

# #### Adding a new column as the length of the name

# In[206]:


data['len']=data['name'].str.len()


# What is the average length of successfull and unsuccessful projects?

# In[207]:


data[['len','state']].groupby('state').mean()


# Length of successful projects and unsuccessful projects seems to be the same.

# In[208]:


data['len'].iplot(kind='histogram',theme='polar',title='Distribution of length of the project name')


# There seems to be no relation between name of the project  and the result. So we will drop the name and len column.

# In[209]:


data.drop('len',axis=1,inplace=True)
col_details.loc['ID']['Action']='delete'
col_details.loc['name']['Action']='delete'
col_details


# >## <a id='CategoryandMainCategory'>Category and Main_Category</a>

# How many categories and main categories are there?

# In[210]:


data['category'].nunique(),data['main_category'].nunique()


# #### Measuring performance of different Main categories.

# In[211]:


y=(data[['main_category','state']].groupby('main_category').mean().sort_values(by='state',ascending=False))*100
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='state',theme='polar',hline=y['state'].mean(),title='Success rate of various main categories')


# #### Measuring performance of diffent categories.

# In[212]:


y=(data[['category','state']].groupby('category').mean().sort_values(by='state',ascending=False))*100
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='state',theme='polar',hline=y['state'].mean(),title='Success rate of various categories')


# There are over 150 different categories which would require a lot of computation so we are going to consider only main category as a factor for our model

# In[213]:


y=cat_feat('main_category')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Main Category')


# In[214]:


col_details.loc['category']['Action']='delete'
col_details.loc['main_category']['Action']='Done'
col_details


# >## <a id='Country'>Country</a>

# In[215]:


y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='occurence rate',theme='polar',title='Country Distribution')


# #### Labelling countries with < 1% contribution as 'Others'

# In[216]:


y=cat_feat('country')
rm=y[y['occurence rate']<1].index
data['country']=data['country'].apply(lambda x:'Others' if x in rm else x)
test['country']=test['country'].apply(lambda x:'Others' if x in rm else x)


# In[217]:


y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='occurence rate',theme='polar',title='Country Distribution')


# In[218]:


y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Country')


# In[219]:


col_details.loc['country']['Action']='done'
col_details


# >## <a id='Currency'>Currency</a>

# In[220]:


y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='count',title='Currency Distribution')


# #### Labelling currencies with < 1% contribution as 'Others'

# In[221]:


y=cat_feat('currency')
rm=y[y['occurence rate']<1].index
data['currency']=data['currency'].apply(lambda x:'Others' if x in rm else x)
test['currency']=test['currency'].apply(lambda x:'Others' if x in rm else x)


# In[222]:


y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='count',title='Currency Distribution')


# Are there people who are not from US but still using USD?

# In[223]:


data[(data['country']!='US')&(data['currency']=='USD')]['country'].value_counts()


# What are the currencies used in this country?

# In[224]:


data[(data['country']=='N,0"')]['currency'].value_counts()


# Does it make any diffence for them to use USD instead of any other currency?

# In[225]:


(data[(data['country']=='N,0"')][['currency','state']].groupby('currency').mean().sort_values(by='state',ascending=False))*100


# #### Occurence rate Vs. Success Rate of currencies

# In[226]:


y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Currency')


# In[227]:


col_details.loc['currency']['Action']='done'
col_details


# >## <a id='DeadlineandLaunched'>Deadline and Launched</a>

# What is the data type of deadline and launched column?

# In[228]:


type(data['deadline'][0]),type(data['launched'][0])


# Deadline and Launched are in string format. We will need to convert them as a datetime object. We will also create a new feature which will tell us the duration for which the fundraiser was active.

# In[229]:


data['launched']=pd.to_datetime(data['launched'])
data['deadline']=pd.to_datetime(data['deadline'])
data['duration']=data[['launched','deadline']].apply(lambda x:(x[1]-x[0]).days,axis=1)
test['launched']=pd.to_datetime(test['launched'])
test['deadline']=pd.to_datetime(test['deadline'])
test['duration']=test[['launched','deadline']].apply(lambda x:(x[1]-x[0]).days,axis=1)


# #### Average duration of fundraising for different main categories.

# In[230]:


y=data[['main_category','duration']].groupby('main_category').mean().sort_values(by='duration',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='duration',theme='polar',hline=y['duration'].mean(),title='Average Duration of main categories')


# #### Average duration of fundraising for different categories.

# In[231]:


y=data[['category','duration']].groupby('category').mean().sort_values(by='duration',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='duration',theme='polar',hline=y['duration'].mean(),title='Average duration of categories')


# #### Taking care of outliers

# What is the maximum and minimum duration of fundraising for any project?

# In[232]:


data['duration'].max(),data['duration'].min()


# Deleting any record for which the duration is more than a year.

# In[233]:


data.drop(data[(data['duration']>365)].index,axis=0,inplace=True)
test.drop(test[(test['duration']>365)].index,axis=0,inplace=True)


# In[234]:


data['duration'].iplot(kind='histogram',theme='polar',title='Duration Distribution')


# There are 92 unique values for duration. We will tranform into interval of 10 days so that they reduce to 10 categories.

# In[235]:


data['duration']=data['duration'].apply(lambda x:(int(x/10)+1)*10)
test['duration']=test['duration'].apply(lambda x:(int(x/10)+1)*10)


# In[236]:


data['duration'].nunique()


# #### Occurence rate vs. Success rate

# In[237]:


y=cat_feat('duration')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Duration')


# #### Taking a look at the trend**

# Extracting year month date and day from the launched date

# In[238]:


data['year']=data['launched'].apply(lambda x:x.year)
data['month']=data['launched'].apply(lambda x:x.month)
data['date']=data['launched'].apply(lambda x:x.day)
data['weekday']=data['launched'].apply(lambda x:x.weekday())


# In[239]:


y=cat_feat('year')
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate & Success Rate year wise')


# In[240]:


y=cat_feat('month')
y.rename(index={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
               7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'},
         inplace=True)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate & Success Rate month wise')


# In[241]:


y=cat_feat('date')
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate vs. Success Rate date wise')


# In[242]:


y=cat_feat('weekday')
y.rename(index={0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'},inplace=True)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate vs. Success Rate day wise')


# In[243]:


y=data[['year','usd_goal_real']].groupby('year').mean()
y.reset_index(inplace=True)
y.iplot(kind='bar',x='year',y='usd_goal_real',theme='polar',title='Avg. goal Year wise')


# In[244]:


y=data[data['year']==2017][['main_category','state']].groupby('main_category').mean()
y.rename(columns={'state':'2017'},inplace=True)
yy=data[data['year']<2017][['main_category','state']].groupby('main_category').mean()
yy.rename(columns={'state':'Prev'},inplace=True)
yyy=pd.concat([yy,y],axis=1)
yyy.reset_index(inplace=True)
yyy.iplot(kind='bar',x='main_category',theme='polar',title='Last year performance of various main categories')


# In[245]:


y=data[data['state']==1][['year','usd_goal_real']].groupby('year').sum()
y.reset_index(inplace=True)
y.iplot(kind='bar',x='year',y='usd_goal_real',theme='polar',title='Money raised year wise')


# In[246]:


data.drop(['year','month','date','weekday'],axis=1,inplace=True)
col_details.loc['launched']['Action']='delete'
col_details.loc['deadline']['Action']='delete'
col_details


# >## <a id='Goal'>Goal</a>

# What is the maximum and minimum goal that anyone has asked for?

# In[247]:


data['usd_goal_real'].max(),data['usd_goal_real'].min()


# How many unique values for goals are there?

# In[248]:


data['usd_goal_real'].nunique()


# Which are the top 5 successful fundraisers?

# In[249]:


data[data['state']==1].sort_values(by='usd_goal_real',ascending=False).head(5)


# #### Distribution of goals for various projects

# In[250]:


plt.figure(figsize=(20,5))
plt.scatter(data.index,data['usd_goal_real'],marker='.',s=10)
plt.title('Goal Distribution')
plt.show()


# 

# #### Average Goal main category wise

# In[251]:


y=data[['main_category','usd_goal_real']].groupby('main_category').mean().sort_values(by='usd_goal_real',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='usd_goal_real',hline=[data['usd_goal_real'].mean()],theme='polar',title='Average goal main category wise')


# #### Average Goal category wise

# In[252]:


y=data[['category','usd_goal_real']].groupby('category').mean().sort_values(by='usd_goal_real',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='usd_goal_real',hline=[data['usd_goal_real'].mean()],theme='polar',title='Average goal category wise')


# #### Average goal of a successful vs an unsuccessful fundraiser

# In[253]:


y=data[data['state']==1][['main_category','usd_goal_real']].groupby('main_category').mean()
y.rename(columns={'usd_goal_real':'Successful'},inplace=True)
yy=data[data['state']==0][['main_category','usd_goal_real']].groupby('main_category').mean()
yy.rename(columns={'usd_goal_real':'Unsuccessful'},inplace=True)
y=pd.concat([y,yy],axis=1)
y.reset_index(inplace=True)
y.iplot(kind='bar',fill=True,x='main_category',barmode='overlay',y=['Unsuccessful','Successful'],color=['red','green'],title='Average goal of successful vs unsuccessful fundraiser')


# #### Bucketing of goals

# Dividing goals into bucket of size 10000

# In[254]:


x=data['usd_goal_real'].apply(lambda x:(int(x/10000))*10000)
x=pd.DataFrame(x.value_counts(),index=x.value_counts().index).sort_index()
x['usd_goal_real'][0:25].iplot(kind='bar',title='Goal Distribution',theme='polar')


# There is a signifcant drop in the frequency of projects with goals > 50k and so we will categorize goals such that there is 1 category for every $1000. Remaining will be put into category number 51.

# In[255]:


data['range']=data['usd_goal_real'].apply(lambda x:(int(x/1000))*1000 if x/1000<=50 else 51000)
test['range']=test['usd_goal_real'].apply(lambda x:(int(x/1000))*1000 if x/1000<=50 else 51000)
y=cat_feat('range')
y.reset_index(inplace=True)
y.iplot(kind='line',x='index',y=['occurence rate','success rate'],title='Goal')


# #### Final Touch

# In[256]:


col_details.loc['goal']['Action']='delete'
col_details.loc['pledged']['Action']='delete'
col_details.loc['backers']['Action']='delete'
col_details.loc['usd pledged']['Action']='delete'
col_details.loc['usd_pledged_real']['Action']='delete'
col_details.loc['usd_goal_real']['Action']='delete'
col_details.loc['state']['Action']='done'
col_details


# >## <a id='DataDeletion'>Data Deletion</a>

# In[257]:


for col in col_details.index:
    if col_details.loc[col]['Action']=='delete':
        data.drop(col,axis=1,inplace=True)
        test.drop(col,axis=1,inplace=True)


# In[258]:


data.head()


# In[259]:


test.head()


# >## <a id='DataConversion'>Convert categorical features into dummy variables</a>

# In[260]:


for col in data:
    if (col!='state'):
        data[col]=data[col].apply(lambda x:col+'_'+str(x))
        test[col]=test[col].apply(lambda x:col+'_'+str(x))
        x=pd.get_dummies(data[col],drop_first=True)
        y=pd.get_dummies(test[col],drop_first=True)
        data=pd.concat([data,x],axis=1).drop(col,axis=1)
        test=pd.concat([test,y],axis=1).drop(col,axis=1)


# In[261]:


data.head()


# In[262]:


test.head()


# # <a id='DataModelling'>Data Modelling</a>

# ## <a id='TrainingsetandTestset'>Splitting data into training and test set</a>

# In[263]:


X=data.drop('state',axis=1)
y=data['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## <a id='DecisionTreeClassifier'>Decision Tree Classifier</a>

# In[264]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions=dtc.predict(X_test)
print(classification_report(y_test,predictions))


# In[265]:


print(confusion_matrix(y_test,predictions))


# ## <a id='RandomForestClassifier'>Random Forest Classifier</a>

# In[266]:


rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)
print(classification_report(y_test,predictions))


# In[267]:


print(confusion_matrix(y_test,predictions))


# ## <a id='LogisticRegression'>Logistic Regression</a>

# In[268]:


lr=LogisticRegression()
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
print(classification_report(y_test,predictions))


# In[269]:


print(confusion_matrix(y_test,predictions))


# # <a id='Conclusion'>Conclusion</a>

# To be continued....

# Feedbacks and Suggestions are welcome.
# 65% accuracy is pretty average for a machine learning model. The model is quite skeptical because if you look at the confusion matrix the number of false negative is quite high.
# I would really like how I can further improve the efficiency of the model.
# Here are a few things that I have already tried:
#     1. KNN (Takes half an hour to execute)
#     2. SVM (Takes half an hour to execute)
#     3. Normalizing range instead of categorizing. (Doesn't make any much difference)
#     4. Considering Category as a feature instead of Main Category. (For an addition 144 columns efficiency improves by 1%)
#     5. Considering a combination of Category and Main category. (Doesn't make any much difference)
#     6. Considering country and currency as a feature. (Doesn't make any much difference)
#                                                        
