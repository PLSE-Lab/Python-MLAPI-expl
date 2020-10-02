#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings  
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## One of the biggest Campaign and CRM problems of banks is "Who Are The Term Deposit Customers"
# ## This Notebook focus on to solve this problem through the Bank Marketing Dataset.

# <img src="https://www.interest.co.nz/sites/default/files/feature_images/td-to-managed-fund-transition.jpg" width=600 height=200>
# 

# ## Introduction<br>
# 
# 
# <img src="https://www.canstar.com.au/wp-content/uploads/2013/07/what-is-a-term-deposit.jpg" width=300 height=100>
# 
# A Term deposit is a deposit that a bank or a financial institurion offers with a fixed rate (often better than just opening deposit account) in which your money will be returned back at a specific maturity time. For more information with regards to Term Deposits please click on this link from Investopedia: https://www.investopedia.com/terms/t/termdeposit.asp
# 
# 
# Today organizations, which hire data scientists are especially interested in job candidate's portfolio. Analysis of organization's marketing data is one of the most typical applications of data science and machine learning. Such analysis will definetely be a nice contribution to the protfolio.
# 
# 
# Finding out customer segments, using data for customers, who subscribed to term deposit. 
# This helps to identify the profile of a customer, who is more likely to acquire the product and develop more targeted marketing campaigns.

# # Attribute Descriptions<br>
# <a id="bank_client_data"></a>
# **age:** (numeric)<br><br>
# **job:** type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br><br>
# **marital:** marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br><br>
# **education:** (categorical: primary, secondary, tertiary and unknown)<br><br>
# **default:** has credit in default? (categorical: 'no','yes','unknown')<br><br>
# **housing:** has housing loan? (categorical: 'no','yes','unknown')<br><br>
# **loan:** has personal loan? (categorical: 'no','yes','unknown')<br><br>
# **balance:** Balance of the individual.<br><br>
# **contact:** contact communication type (categorical: 'cellular','telephone') <br><br>
# **month:** last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br><br>
# **day:** last contact day of the month (numeric: 1,2,3,....29,30)<br><br>
# **duration:** last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br><br>
# **campaign:** number of contacts performed during this campaign and for this client (numeric, includes last contact)<br><br>
# **pdays:** number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br><br>
# **previous:** number of contacts performed before this campaign and for this client (numeric)<br><br>
# **poutcome:** outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br><br>
# **deposit** - has the client subscribed a term deposit? (binary: 'yes','no')

# # General View

# In[ ]:


df=pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")
df.info()


# Good news there is no missing value

# In[ ]:


df.head()


# In[ ]:


df.describe()


# # Attributes in Detail

# ## Attribute - Age

# In[ ]:


df_age=df[["age","deposit"]]
df_age.describe()


# In[ ]:


sns.violinplot(x="deposit",y="age",data=df_age)
plt.show()


# In[ ]:


plt.figure(figsize=(17,10))
sns.countplot("age",data=df,hue="deposit")
plt.show()


# In[ ]:


df["age_bin"]=pd.cut(df.age,bins=[18,29,40,50,60,100],labels=['young','midAge','Adult',"old",'Elder'])
sns.countplot(x="age_bin",data=df,hue="deposit")
plt.show()


# * For older than 60 age and younger than 30 age customers term deposit ratio dramatically increasing.
# * Prabably all models will choose age as important attribute

# ## Attribute - Job

# In[ ]:


df_job=df[["job","deposit"]]
df_job.describe()


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x="job",hue="deposit",data=df)
plt.show()


# * Retired and student job categories have the highest deposit customer rates. 

# ## Attribute - Marital

# In[ ]:


df_marital=df[["marital","deposit"]]
df_marital.describe()


# In[ ]:


sns.countplot(x="marital",hue="deposit",data=df)
plt.show()


# * The single category with the highest deposit rate. Singles has money to save :)
# * Lowest one is the married category. Probably most of the people in married category have children :)

# In[ ]:


# Notice how divorced have a considerably low amount of balance.
fig = ff.create_facet_grid(
    df,
    x='duration',
    y='balance',
    color_name='marital',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    colormap={'single': 'rgb(165, 242, 242)', 'married': 'rgb(253, 174, 216)', 'divorced': 'rgba(201, 109, 59, 0.82)'}
)

iplot(fig, filename='facet - custom colormap')


# ## Attribute - Education

# In[ ]:


sns.countplot(x="education",hue="deposit",data=df)
plt.show()


# ## Attribute - Default

# In[ ]:


sns.countplot(x="default",hue="deposit",data=df)
plt.show()


# * Default is one of the most important variables for analysing credit customers. But in this dataset its look like no relation to deposit

# ## Attribute - Housing

# In[ ]:


sns.countplot(x="housing",hue="deposit",data=df)
plt.show()


# * It is an expected result that those with home loans do not have deposits :)
# * Because these people have chosen to secure their future as homeowners instead of deposits.
# * Probably it is most correlated variable

# ## Attribute - Loan

# In[ ]:


sns.countplot(x="loan",hue="deposit",data=df)
plt.show()


# * Loan category not as much correlated with deposit as housing.

# ## Attribute - Balance

# In[ ]:


sns.boxplot(x="deposit",y="balance",data=df)
plt.show()


# * we need to understand distribution of balance and categorize it.

# In[ ]:


b_df = pd.DataFrame()
b_df['balance_yes'] = (df[df['deposit'] == 'yes'][['deposit','balance']].describe())['balance']
b_df['balance_no'] = (df[df['deposit'] == 'no'][['deposit','balance']].describe())['balance']
b_df.drop(['count', '25%', '50%', '75%']).plot.bar(title = 'Balance and deposit statistics')
plt.show()


# In[ ]:


df["balance_cat"] = np.nan
df.loc[df['balance'] <0, 'balance_cat'] = 'negative'
df.loc[(df['balance'] >=0)&(df['balance'] <=5000), 'balance_cat'] = 'low'
df.loc[(df['balance'] >5000)&(df['balance'] <=20000), 'balance_cat'] = 'mid'
df.loc[(df['balance'] >20000), 'balance_cat'] = 'high'
sns.countplot(x="balance_cat",hue="deposit",data=df)
plt.show()


# ## Attribute - Contact

# In[ ]:


sns.countplot(x="contact",hue="deposit",data=df)
plt.show()


# ## Attribute - Month

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="month",hue="deposit",data=df,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
plt.show()


# * Looks like monthly effect on feb,mar,apr,sep,oct,dec. 
# * May be we can group some months on feature engineering part. 

# In[ ]:


df["month_bins"]=pd.cut(df.day,bins=4,labels=["q1","q2","q3","q4"])
plt.figure(figsize=(10,5))
sns.countplot(x="month_bins",hue="deposit",data=df)
plt.show()


# ## Attribute - Day

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="day",hue="deposit",data=df)
plt.show()


# In[ ]:


df["day_cat"] = np.nan
df.loc[df['day'] <5, 'day_cat'] = '1'
df.loc[(df['day'] >=5)&(df['day'] <=9), 'day_cat'] = '2'
df.loc[(df['day'] >=10)&(df['day'] <=13), 'day_cat'] = '3'
df.loc[(df['day'] >=14)&(df['day'] <=21), 'day_cat'] = '4'
df.loc[(df['day'] >=22), 'day_cat'] = '5'
plt.figure(figsize=(10,5))
sns.countplot(x="day_cat",hue="deposit",data=df)
plt.show()


# In[ ]:


df["day_bins"]=pd.cut(df.day,bins=4,labels=["w1","w2","w3","w4"])
plt.figure(figsize=(10,5))
sns.countplot(x="day_bins",hue="deposit",data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.swarmplot(x="month",y="day",hue="deposit",data=df,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
plt.show()


# * That's great view. I think day and month are important attributes.
# * Potential clients opted to suscribe term deposits during the seasons of fall and winter. The next marketing campaign should focus its activity throghout these seasons.

# ## Attribute - Campaign

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="campaign",hue="deposit",data=df)
plt.show()


# In[ ]:


df["campaign_cat"] = np.nan
df.loc[df['campaign'] ==1, 'campaign_cat'] = 0
df.loc[(df['campaign'] >1), 'campaign_cat'] = 1
sns.countplot(x="campaign_cat",hue="deposit",data=df)
plt.show()


# ## Attribute - Pdays

# In[ ]:


plt.figure(figsize=(10,5))
df["pdays_bin"]=pd.cut(df.pdays,bins=3,labels=["c1","c2","c3"])
sns.countplot(x="pdays_bin",hue="deposit",data=df)
plt.show()


# ## Attribute - Previous

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="previous",hue="deposit",data=df)
plt.show()


# In[ ]:


df["previous_cat"] = np.nan
df.loc[df['previous'] <=2, 'previous_cat'] = 0
df.loc[(df['previous'] >2), 'previous_cat'] = 1
sns.countplot(x="previous_cat",hue="deposit",data=df)
plt.show()


# ## Attribute - Poutcome

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="poutcome",hue="deposit",data=df)
plt.show()


# # DATA PREPARING FOR ML

# In[ ]:


df.head()


# * We must drop Duration and convert categorical variables to numeric.

# In[ ]:


df.drop(labels = ['duration'], axis = 1, inplace = True)
df['deposit']=df['deposit'].map({'yes':1,'no':0})
df = pd.get_dummies(df, columns=['job','marital','education',"month",'default','housing',"loan","contact","poutcome","age_bin","balance_cat","pdays_bin","day_cat","day_bins","month_bins"])
cor_deposit=df.corr()
cor_deposit["deposit"].sort_values(ascending=False)


# In[ ]:


x_train=df.drop(labels=['deposit'],axis=1)
y_train=df['deposit'].astype(int)
X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.25, random_state=2)

kfold = StratifiedKFold(n_splits=10)

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier(random_state = random_state))
classifiers.append(LGBMClassifier(random_state = random_state))
classifiers.append(CatBoostClassifier())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis",'XGBClassifier','LGBMClassifier','CatBoostClassifier']})
plt.figure(figsize=(20,10))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
plt.axvline(0.74)
plt.axvline(0.72)
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:


cb = CatBoostClassifier()
cb.fit(X_test,Y_test)

y_pred=cb.predict(X_test)
y_true=pd.DataFrame(Y_test)
from sklearn.metrics import classification_report
cr=classification_report(y_true,y_pred,output_dict=True)
pd.DataFrame(cr)


# In[ ]:


plt.figure(figsize=(20,5))
fi=cb.get_feature_importance(prettified=True).head(10)
sns.barplot(x="Feature Id",y="Importances",data=fi)
score=cb.score(X_test,Y_test)
plt.title('Accuracy: '+str(score))
plt.show()


# 
# 
# <img src="https://media1.tenor.com/images/9bb0a3b1cb26bc09de21d61e61c37241/tenor.gif?itemid=7263498" width=150 height=50>

# # Conclusion
# 
# * The people who younger than 30 and older than 60 years old most likely to subscribe for term deposit.
# * Age attribute confirmed with job attribute. Because Retired and Students are most likely to subscribe for term deposit.
# * Customers with 'blue-collar' and 'services' jobs are less likely to subscribe for term deposit.
# * Divorced have a considerably low amount of balance.
# * Education significant impact the amount of balance.
# * Customers are more willing to invest either before 8th or after 23rd of the month.
# * Potential clients opted to suscribe term deposits during the seasons of fall and winter.
# *  Potential clients in the low balance and no balance category were more likely to have a house loan than people in the average and high balance category. What does it mean to have a house loan? This means that the potential client has financial compromises to pay back its house loan and thus, there is no cash for he or she to suscribe to a term deposit account. However, we see that potential clients in the average and hih balances are less likely to have a house loan and therefore, more likely to open a term deposit. Lastly, the next marketing campaign should focus on individuals of average and high balances in order to increase the likelihood of suscribing to a term deposit.
# * Target individuals with a higher duration (above 375): Target the target group that is above average in duration, there is a highly likelihood that this target group would open a term deposit account. The likelihood that this group would open a term deposit account is at 78% which is pretty high. This would allow that the success rate of the next marketing campaign would be highly successful.
# 
