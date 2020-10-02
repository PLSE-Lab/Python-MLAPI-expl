#!/usr/bin/env python
# coding: utf-8

# # Titanic_EDA,Visualization, ML

# ![image.png](attachment:image.png)

# This is my first notebook in kaggle, so please **upvote** if you like it and leave a comment if you have any advice. I will appreciate that. There's totally 5 sections in this notebook. By Tableau and seaborn, we can do the EDA. By statistical analysis on the features and building machine learning models, we can make a prediction of whether a person survive or not.

# -----
# ## Table of Content
# *   [1. Import Data and Data Structure](#1)  
#     *    [ 1.1 Import Data and Package](#2)    
#     *    [ 1.2 Overview of Data Structure (Tableau)](#3)  
#     
#     
# *   [2. Data Visualization and Missing Values](#4)  
#     *       [2.1 Categorical Encoding](#5)
#     *     [2.2 Overview of Relation](#15)
#     *       [2.3 Visualization of Features](#6)
#     *    [ 2.4 Missing Values](#7)     
#     
#     
# *   [3. Feature Engineering and Statistical Analysis](#8)  
#     * [3.1 Features Generation](#9)
#     * [3.2 Statistical Analysis and Feature Selection ](#10)    
#     
# 
# *   [4. Modelling](#11)
#     * [4.1 Building Machine Learning Models (KNN, XGBoostClassifier, LGBMClassfier etc.)](#12)
#     * [4.2 Model Comparison ](#13)  
#     
# 
# *   [5. Data Submission](#14)    

#  <a id="1"></a> 
#  # 1. Import and Read Data

# <a id="2"></a> 
# ## 1.1 Import Data and Package

# In[ ]:


# read the data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Pearson Residual Test
import scipy.stats as stats
from scipy.stats import chi2_contingency

# Machine Learning Model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB

# import Tableau
from IPython.display import IFrame


file_path1 = "../input/titanic/train.csv"
file_path2= "../input/titanic/test.csv"
data = pd.read_csv(file_path1)
test_data=pd.read_csv(file_path2)
data.describe()


# <a id="3"></a> 
# ## 1.2 Overview of Data structure 
# > Find out the data structure and type

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1592943474846' style='position: relative'>\n<noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ta&#47;Task3_15929207496170&#47;Dashboard2&#47;1_rss.png' style='border: none' />\n</a></noscript><object class='tableauViz'  style='display:none;'>\n<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />\n<param name='embed_code_version' value='3' /> <param name='site_root' value='' />\n<param name='name' value='Task3_15929207496170&#47;Dashboard2' />\n<param name='tabs' value='no' /><param name='toolbar' value='yes' />\n<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ta&#47;Task3_15929207496170&#47;Dashboard2&#47;1.png' />\n<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' />\n<param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />\n<param name='display_count' value='yes' /><param name='language' value='en' />\n<param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1592943474846');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='2127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


# show the first five line of data
data.head()


# In[ ]:


# show the data struture and type
data.info()


# In[ ]:


test_data.info()


# There are totally 11 variables and we can divide them into 3 types: categorical variable, numeric variables and text variable:
# ### 1.2.1 Categorical Variable
# 
# * **Pclass **:
# >         1=1st  
# >         2=2st             
# >         3=3st 
# 

# In[ ]:


data.Pclass.value_counts()
#s1=data.groupby('Pclass').apply(lambda df: df.loc[df.Survived==0].Survived.value_counts())
#s2=data.groupby('Pclass').Survived.count()
s1=data.groupby('Pclass').apply(lambda df: df.Survived.value_counts()/len(df)) 
s2=data.groupby('Pclass').apply(lambda df: df.Survived.value_counts()) 
pd.concat([s2,s1],axis=1,keys=['Survived/Death count','Survived/Death Rate'])


# The above table shows the Survival rate for each Pclass which is an decreasing trend: 0.629 > 0.472 > 0.242 as 'Pclass' value increasing. Also Pclass=3 has the highest death population. Pclass will be an important feature for Survived prediction.
# * **Sex**:
# >         Male 
# >         Female  
# 
# Most of the passengers are male and female passenger was less than 50%

# In[ ]:


data.Sex.value_counts()


# * **Cabin**: 
# 
# The cabin number is a character followed by a number and there are 147 different cabin number in train dataset and 76 different numbe in test dataset. Now we group these Cabin number by their first character.
# 
# 
# 

# In[ ]:


data.Cabin.value_counts()
test_data.Cabin.value_counts()


# In[ ]:


data.Cabin.str[0].value_counts()
test_data.Cabin.str[0].value_counts()


# Now we can get all the types of Cabin are starting with these character:
# >         A 
# >         B  
# >         C 
# >         D  
# >         E 
# >         F  
# >         G 
# >         T  
# 
# * Embarked (Port of Embarkation):
# >         S 
# >         C  
# >         Q 
# 
# ------------
# 

# In[ ]:


s3=pd.Series(data.Embarked.value_counts())

pd.concat([s3,s3/889],axis=1,keys=['count(Embarked)','portion'])


# ### Numeric Variables
# * PassengerId (Uniquely define a passenger)

# In[ ]:


data.PassengerId.value_counts()


# There's 891 person in train dataset and each person has a unique PassengerId. However, PassengerId is a numeric value and there's a pretty small correlation (-0.05) between 'Survived' and 'PassengerId'. We may not consider this feature in the prediction modelling.
# 
# 
# * Age
# * SibSp (# of siblings / spouses aboard the Titanic)
# * Parch (# of parents / children aboard the Titanic)
# * Fare (the ticket price)
# 
# -------
# ### Text Variable
# * Name
# * Ticket
# ----------

# -------
# <a id="4"></a>
# # 2. Data Visualization and Missing Values
# <a id="5"></a> 
# ## 2.1 Categorical Encoding
# Apply the categorical encoding method to the categorical variable and visualize their relation
# 

# In[ ]:


copy1=data.copy()
copy2=test_data.copy()


# 
# ### Sex
# * Convert Sex into categorical variable by applying Label-Encoding

# In[ ]:


label_encoder=LabelEncoder()
copy1['new_Sex']=label_encoder.fit_transform(copy1['Sex'])
copy2['new_Sex']=label_encoder.transform(copy2['Sex'])


# ### Cabin
# * Oragnize the Cabin data by extracting their first character (A, B, C, D, E, F, G, T)
# * Treat them as categorical variable

# In[ ]:


copy1['new_Cabin']=data['Cabin'].str[0]
s1=pd.Series(copy1.new_Cabin.value_counts())
s2=s1/91
pd.concat([s1,s2],axis=1,keys=['count(new_cabin)','portion'])
# Copy the test dataset and add a new column


# In[ ]:



copy2['new_Cabin']=test_data['Cabin'].str[0]
s3=pd.Series(copy2.new_Cabin.value_counts())
pd.concat([s3,s3/76],axis=1,keys=['count(new_Cabin)','portion'])


# * new_Cabin='C' has the highest portion in both dataset
# * Treat them as categorical variable by applying Label_Encoding
# * However, 'new_Cabin' contains missing value (null)
# * Regard all the missing value as new_Cabin='Z' for now and leave this problem to next part

# In[ ]:


copy1['new_Cabin']=copy1['new_Cabin'].fillna("Z")
copy2['new_Cabin']=copy2.new_Cabin.fillna("Z")

label_encoder=LabelEncoder()
copy1['new_Cabin']=label_encoder.fit_transform(copy1['new_Cabin'])
copy2['new_Cabin']=label_encoder.transform(copy2['new_Cabin'])


# ### Embarked
# * Embarked also contains missing value
# * Regard all the missing value as Embarked='N'
# * Transfer Embarked into integer by applying label encoder

# In[ ]:


copy1['new_Embarked'] = copy1['Embarked'].fillna("N")

label_encoder=LabelEncoder()
copy1['new_Embarked']=label_encoder.fit_transform(copy1['new_Embarked'])


# ----
# <a id="15"></a> 
# ## 2.2 Overview of Relation
# Visualize the relation between feature by correlation Heatmap and pairs plot

# * Apply the pairs plot

# In[ ]:


sns.pairplot(data,hue='Pclass', diag_kws={'bw':0.1}, palette="husl")


# * Apply the correlation Heatmap

# In[ ]:


# the correlation matrix
features=['PassengerId','Survived','Pclass','new_Sex','Age','SibSp','Parch','Fare','new_Cabin','new_Embarked']
corr=copy1[features].corr() 
# mask the upper triangle
sns.set(style="white")
plt.figure(figsize=(11,7))
mask=np.triu(np.ones_like(corr,dtype=np.bool))
# colour
cmap=sns.diverging_palette(240,10,n=9)
# annot to display the value
sns.heatmap(corr,annot=True,mask=mask,cmap='RdYlBu',linewidths=0.6)


# * Some of the features are highly correlated with Survived
# * Apply the pairs plot on those features

# ------
# <a id="6"></a> 
# ## 2.3 Visualization of Features
# Some of the features are highly correlated with survival rate or with each other. Data visualization can help us find out the pattern behind them.
# 
# ### Gender, Age and Survived
# * Firstly, Consider the Age structure of each Gender 
# *  We can discover that the Age structure of male and female are similar

# In[ ]:


s1=copy1.loc[copy1.Sex=='female'].Age.describe()
s2=copy1.loc[copy1.Sex=='male'].Age.describe()
pd.concat([s1,s2],axis=1,keys=['Age|Sex=female','Age|Sex=male'])


# * The correlation Heatmap shows that Gender is highly correlated to 'Survived'
# * Calculate the survival rate for men and women

# In[ ]:


s1=copy1.groupby('Sex').apply(lambda df: df.Survived.value_counts())
s2=copy1.groupby('Sex').apply(lambda df: df.Survived.value_counts()/len(df))
s3=pd.concat([s1,s2],axis=1,keys=['count(Survival/Death)','Survival/Death rate'])
s3


# In[ ]:


#fig, ax = plt.subplots(figsize=(12,5),ncols=2)
#d1=copy1.loc[copy1.Sex=='male']
#d2=copy1.loc[copy1.Sex=='female'].Survived.value_counts()

#d1=copy1.loc[copy1.Sex=='female']
#f=['Survived']
#X=d1[f].Survived.value_counts()

sns.set(style="whitegrid")
ax1=sns.barplot(y=s3.index, x =s3['count(Survival/Death)'],linewidth=2.5,facecolor=(1,1,1,0),errcolor="1", edgecolor=".1")
ylabels = ['(female, survived)','(female, Dead)', '(male, Dead)', '(male, survived)']
ax1.set_yticklabels(ylabels)
i=0
list1=s3['Survival/Death rate']
for p in ax1.patches:
    label = list1[i]*100
    i=i+1
    plt.text(-36+p.get_width(), p.get_y()+0.55*p.get_height(),
             str('{:1.2f}'.format(label))+'%',
             ha='center', va='center')


# * Obviously, survival rate of female is much higher than male
# * Consider Sex vs. Age vs. Survival

# In[ ]:


sns.set(style="darkgrid")
f, (ax1,ax2) = plt.subplots(figsize=(18,7),ncols=2)
s1= copy1.loc[copy1.Sex=='female']
ax1 = sns.distplot(a=s1.Age,bins=34, kde=False, 
                  hist_kws={"rwidth":1,'edgecolor':'black', 'alpha':1.0},color='azure',label="Age",ax=ax1)
s2= s1.loc[(s1.Survived==1)]
ax1 = sns.distplot(a=s2.Age,bins=34, kde=False, 
                  hist_kws={"rwidth":1,'edgecolor':'black', 'alpha':1.0},color='cyan',ax=ax1)
#ax1.set_title("Histogram of Survival, female")
ax1.legend(['total count', 'Survived count'])
ax1.set_title("Histogram of Survival, female")


s3= copy1.loc[copy1.Sex=='male']
ax2 = sns.distplot(a=s3.Age,bins=34, kde=False, 
                  hist_kws={"rwidth":1,'edgecolor':'black', 'alpha':1.0},color='lavender',label="Age",ax=ax2)
s4= s1.loc[(s1.Survived==1)]
ax2 = sns.distplot(a=s4.Age,bins=34, kde=False, 
                  hist_kws={"rwidth":1,'edgecolor':'black', 'alpha':1.0},color='orchid',ax=ax2)
#ax1.set_title("Histogram of Survival, female")
ax2.legend(['total count', 'Survived count'])
ax2.set_title("Histogram of Survival, male")


# * The portion of blue area is much larger than purple
# * At any range of age, the survival rate of female is much higher than male
# 
# ### Parch and SibSp vs. Age
# * SibSp is highly correlated with SibSp
# * There's negative correlation between Parch and SibSp vs. Age
# * Group the dataset by pairs of (x=Parch,y=SibSP) value and compare each group's average age
# * Ignore the 'Age' missing value in this part

# In[ ]:


copy3=copy1.loc[copy1.Age.notnull()]


# In[ ]:


X=[0.0,1.0,2.0,3.0,4.0,5.0,6.0]
Y=[0,1,2,3,4,5]
def f(x,y):
    a=copy3.loc[(copy3.Parch==x)&(copy3.SibSp==y)].Age.mean()
    return a
X,Y=np.meshgrid(X,Y)
#Z=f(X,Y)
Z=np.zeros((6, 7))
for i in range(6):
    for j in range(7):
        Z[i][j]=f(X[i][j],Y[i][j])
a= copy3.loc[(copy3.SibSp>=4)].Age.mean()
for i in range(2):
    for j in range(7):
        Z[i+4][j]=a

Z[0][6]=Z[0][5]
Z[2][4]=Z[2][5]=Z[2][6]=Z[2][3]
Z[3][3]=Z[3][4]=Z[3][5]=Z[3][6]=Z[3][2]


# * 3D surface plot and contour plot to visualize the relation among SibSp, Parch and Age
# * Z represents the average Age for certain pairs of (SibSp,Parch) value

# In[ ]:


#Z = np.cos(X ** 2 + Y ** 2)
fig= plt.figure(figsize=(10,6))
#ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title('Average Age for SibSp and Parch')
ax.set_xlim(0, 6);
ax.set_ylim(5, 0);
ax.set_xlabel('Parch')
ax.set_ylabel('SibSp')
ax.set_zlabel('Age.mean()');
#plt.show()
plt.show()


# In[ ]:



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()

cset = ax.contourf(X, Y, Z)
plt.colorbar(cset)

ax.set_title('contour plot');
plt.show()


# * Larger SibSp and Parch value implies lower average Age
# * Young people are more likely to be companied by their family member
# 

# ### Age, Family Size, Fare and Survive

# * Age and Family size vs Survive
# * Divide all the people into several Age group: 0~18, 18~35, 35~55, 55~80

# In[ ]:


copy3=copy1[['Age','Fare','Survived','SibSp','Parch']]
copy3['family_size']=copy3['SibSp']+copy3['Parch']
def f (p):
    if p<= 18: return '0~18'
    if p>18 and p<35: return '18~35'
    if p>=35 and p<55: return '35~55'
    if p>=55: return '55~80'
copy3['age_group']=copy3.Age.map(lambda p: f(p))


# In[ ]:



sns.set(style="darkgrid")
#sns.set(style="darkgrid")
sns.catplot(x='family_size',col='age_group', col_wrap=2,data=copy3,hue="Survived", kind="count",height=8, aspect=.8,palette=['aqua','pink'])
#sns.catplot(x='Age',y='SibSp',data=copy3,height=8, aspect=.7)


# * Visualize Age, Fare,Family size and Survived

# In[ ]:


fig = plt.figure()
fig, ax = plt.subplots(figsize=(11,6))
ax = fig.add_subplot(111, projection='3d')
X=copy1['SibSp']+copy1['Parch']
Y=copy1.Age
Z=copy1.Fare
g=ax.scatter(X, Y, Z,  c= copy1.Survived,marker='o',cmap='cool')
plt.colorbar(g)
ax.set_zlim(0, 250);
ax.set_xlabel('Family size')
ax.set_ylabel('Age')
ax.set_zlabel('Fare');
plt.show()


# * Fare of Purple dots are higher than blue dots on average
# * Higher Fare price tends to stay alive

# ### Pclass, Fare vs. Survived
# * First, consider Pclass and Fare which are negatively correlated
# * Fare distribution given a certain Pclass

# In[ ]:


s1=copy1.loc[copy1.Pclass==1].Fare.describe()
s2=copy1.loc[copy1.Pclass==2].Fare.describe()
s3=copy1.loc[copy1.Pclass==3].Fare.describe()
pd.concat([s1,s2,s3],axis=1,keys=['Fare|Pclass=1','Fare|Pclass=2','Fare|Pclass=3'])


# In[ ]:


sns.set(style="darkgrid")
fig, ax1 = plt.subplots(figsize=(16,5))
ax=sns.kdeplot(data=copy1.loc[copy1.Pclass==1]['Fare'],shade=True,color='red')
ax.set_title("Fare distribution|Pclass")
ax=sns.kdeplot(data=copy1.loc[copy1.Pclass==2]['Fare'],shade=True,color='blue')
ax=sns.kdeplot(data=copy1.loc[copy1.Pclass==3]['Fare'],shade=True,color='purple')
ax.legend(["Fare|Pclass=1","Fare|Pclass=2","Fare|Pclass=3"])


# 
# * Acccoring to the above distribution plot and table, we can roughly say that Fare|Pclass=1> Fare|Pclass=2 > Fare|Pclass=3 on average.
# * The Fare|Pclass=1 also has a larger standard error than any other Pclass groups
# * It implies that as Pclass get upper, the ticket price will be more expensive on average

# In[ ]:


fig, ax1 = plt.subplots(figsize=(12,15))
sns.swarmplot(x=copy1.Survived,y=copy1.Fare,hue=copy1.Pclass, palette='cool')


# In[ ]:


sns.catplot(x="Survived",y="Fare",hue="Pclass",data=copy1,kind='violin',height=8, aspect=2, palette=['violet','turquoise','tomato'])


# ### Sex, Pclass, Fare and Survived

# In[ ]:


sns.set(style="darkgrid")
sns.catplot(x="Survived",y="Fare",col="Pclass",data=copy1 ,hue="Sex",height=8, aspect=.7, palette=['violet','turquoise'])


# * Higher Fare price and upper class tends to live 

# ### Pclass,Age,Sex vs Survived

# In[ ]:


g = sns.FacetGrid(copy1,height=5, col="Pclass", row="Sex", margin_titles=True, hue = "Survived" )
g = g.map(sns.distplot, "Age",kde=False,bins=15,hist_kws={"rwidth":1,'edgecolor':'black', 'alpha':1.0}).add_legend();


# ### Embarked, Fare and Pclass
# * Embarked and Pclass

# In[ ]:


s0=copy1.Embarked.value_counts()
s1=copy1.loc[copy1.Pclass==1].Embarked.value_counts()
s2=copy1.loc[copy1.Pclass==2].Embarked.value_counts()
s3=copy1.loc[copy1.Pclass==3].Embarked.value_counts()
pd.concat([s0,s1/s0*100,s2/s0*100,s3/s0*100],axis=1,keys=['Total','Embarked|Pclass=1 (%)','Embarked|Pclass=2 (%)','Embarked|Pclass=3 (%)'])


# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x=copy1.Embarked,y=copy1.Fare,hue=copy1.Pclass,palette='cool')


# ### Embarked, Pclass and Fare vs. Survived
# * Embarked and Survived

# In[ ]:


plt.figure(figsize=(12,12))
sns.swarmplot(x=copy1.Embarked,hue=copy1.Survived,y=copy1.Fare,palette='spring')


# * Embarked S and C have higher survival rate than Embarked Q
# 

# In[ ]:


sns.set(style="darkgrid")
sns.catplot(x="Survived",col="Pclass",data=copy1 ,hue="Embarked",kind='count',height=8, aspect=.7, palette=['gold','orangered','brown'])


# ------
# <a id="7"></a> 
# ## 2.4  Missing Values
# There are totally 891 entries in the train dataset and 418 entries in the test dataset. However, the variable: Age, Cabin and Embarked in train dataset contain null values and the variables:  Age, Fare,Cabin in test dataset contain null values. Find the best method to deal with missing value problem.

# ### Embarked
# * According to the correlation heatmap, we know that Embarked is highly correlated to Sex, Fare and Pclass
# * We gonna find out the missing values based on these 2 features

# In[ ]:


copy1.loc[copy1.Embarked.isnull()]


# * There's only 2 missing values in Embarked
# * They share the same Sex,Pclass and Fare

# In[ ]:


copy1.loc[(copy1.Pclass==1) & (copy1.Sex=='female')].Embarked.value_counts()


# * People from Pclass=1 and Sex= female are more likely to from Embarked=S or C
# * Let's see the Fare description for (Pclass=1 & Sex='female')

# In[ ]:


s1=copy1.loc[(copy1.Pclass==1) & (copy1.Sex=='female')]
s2=s1.loc[s1.Embarked=='S'].Fare.describe()
s3=s1.loc[s1.Embarked=='C'].Fare.describe()
pd.concat([s2,s3],axis=1,keys=['Fare|Embarked=S', 'Fare|Embarked=C'])


# * As the mean of Fare are 99 and 115 for Embarked='S' and Embarked='C' respectively
# * 99 is much closer to 80 than 115
# * Assign Embarked='S' to these 2 missing values

# * Refill the missing values for new_Embarked in both training and testing dataset

# In[ ]:


copy1['Embarked']=copy1.Embarked.fillna('S')


# In[ ]:


label_encoder=LabelEncoder()
copy1['new_Embarked']=label_encoder.fit_transform(copy1['Embarked'])
copy2['new_Embarked']=label_encoder.transform(copy2['Embarked'])


# 
# ### Cabin
# * Cabin is highly correlated to Pclass, Fare, Age and Sex
# 

# In[ ]:


# make a temporary column to store the first character
print('Null Pertcentage of Cabin in tranining dataset : ',str(len(copy1.loc[copy1.Cabin.isnull()])/len(copy1.Cabin)*100),'%')
print('Null Pertcentage of Cabin in testing dataset : ',str(len(copy2.loc[copy1.Cabin.isnull()])/len(copy2.Cabin)*100),'%')


# * 77% of the Cabin data are null for both dataset
# * Cabin column should be abandoned

# In[ ]:


copy1=copy1.drop(['Cabin','new_Cabin'],axis=1)
copy2=copy2.drop(['Cabin','new_Cabin'],axis=1)


# ### Age
# * Age is highly correlated to Pclass, SibSp and Parch
# * We can apply a prediction for the missing values based on these features
# * Gonna choose LogisticRegression in this part

# In[ ]:




# training dataset
feature=['Pclass','SibSp','Parch']
copy3=copy1.loc[copy1.Age.notnull()]
x_train=copy3[feature]
#convert y_train to integer
y_train=copy3.Age.astype(int)

# prediction
copy4=copy1.loc[copy1.Age.isnull()]
x_test=copy4[feature]

log=LogisticRegression()
log.fit(x_train,y_train)
y_pred = log.predict(x_test)


# In[ ]:


# assign the new values to Age column
copy1.loc[copy1.Age.isnull(), "Age"] = y_pred


# In[ ]:


# Testing datset
x_test=copy2.loc[copy2.Age.isnull()][feature]
y_pred=log.predict(x_test)
copy2.loc[copy2.Age.isnull(), "Age"] = y_pred


# ### Fare
# * There's a missing entry in Fare column of testing dataset
# * Fare is highly correlated to Pclass, new_Embarked,new_Sex, SibSP and Parch
# * Apply the random forest regressor to make a prediction

# In[ ]:


copy2.loc[copy2.Fare.isnull()]


# In[ ]:


# training dataset
feature=['Pclass','SibSp','Parch','new_Sex','new_Embarked']
copy3=copy2.loc[copy2.Fare.notnull()]
x_train=copy3[feature]
#convert y_train to integer
y_train=copy3.Fare

# prediction
copy4=copy2.loc[copy2.Fare.isnull()]
x_test=copy4[feature]

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(x_train,y_train)
y_pred = forest_model.predict(x_test)


# In[ ]:


copy2.loc[copy2.Fare.isnull(), "Fare"] = y_pred


# -----
# <a id="8"></a> 
# # 3. Feature Engineering and Statistical Analysis
# In this part, we will create some features by applying feature engineering. Then select the useful features among them by statistical analysis to prepare for model prediction.

# <a id="9"></a> 
# ## 3.1 Features Generation
# ### Name
# * The name of passengers also contain title
# * Extract the title from name and treat it as categorical variable

# In[ ]:


#[i.split('.')[1] for i in data.Name]
#for i in range(len(data.Name)):
    #title = data.Name[i].split('.')[0]
    #title = title.split(',')[1]
copy1['Title']=[n.split('.')[0] for n in copy1.Name]
copy1['Title'] = [t.split(',')[1] for t in copy1.Title]

copy2['Title']=[n.split('.')[0] for n in copy2.Name]
copy2['Title'] = [t.split(',')[1] for t in copy2.Title]


# * Demonstrate all sorts of title:
# * However, some of the feature contains only 1 value that will cause problem: For example, there's only 4 'Col' in the whole dataset. If all the 'col' in training set are dead then the estimating 'col' survival rate will be 0, the prediction in testing set will absolutely be dead. This kind of prediction is probably inaccurate.

# In[ ]:


pd.concat([copy1,copy2]).Title.value_counts()
#copy1.Title.value_counts()


# * Convert 'Title' column into category variables
# * There are 18 different kinds of titles in dataset and train dataset only contains 17 of them
# * Fit the Label-Encoding on concatation of train dataset and test datset

# In[ ]:


def transform(i):
    if 'Mr' in i: return 'Mr'
    if 'Mrs'in i: return 'Mrs'
    if 'Miss'in i: return 'Miss'
    if 'Master' in i: return 'Master'
    else: return 'NA'
    
copy1['Title']=[transform(i) for i in copy1.Title]
copy2['Title']=[transform(i) for i in copy2.Title]
label_encoder=LabelEncoder()
label_encoder.fit_transform(pd.concat([copy1,copy2])['Title'])
copy1['new_Title']=label_encoder.transform(copy1['Title'])
copy2['new_Title']=label_encoder.transform(copy2['Title'])


# ### Ticket
# * Create a feature Ticket's string length

# In[ ]:


copy1['Ticket_length']=[len(i) for i in copy1.Ticket]
copy2['Ticket_length']=[len(i) for i in copy2.Ticket]


# ### SibSp & Parch
# * SibSp and Parch are both illustrating number of family members
# * Set a new feature Famsize = SibSp + Parch

# In[ ]:


copy1['Famsize']=copy1['SibSp']+copy1['Parch']
copy2['Famsize']=copy2['SibSp']+copy2['Parch']


# <a id="10"></a> 
# ## 3.2 Statistical Analysis and Feature Selection
# * Draw the Correlation Heatmap based on the features in hand

# In[ ]:


# the correlation matrix
corr=copy1.corr() 
# mask the upper triangle
sns.set(style="white")
plt.figure(figsize=(11,7))
mask=np.triu(np.ones_like(corr,dtype=np.bool))
# colour
cmap=sns.diverging_palette(240,10,n=9)
# annot to display the value
sns.heatmap(corr,annot=True,mask=mask,cmap='jet',linewidths=0.6)


# * Some of the features have almost no correlation with Survived (eg. PassengerId)
# * The correlation magnitude of [PassengerId, Age, SibSp, Parch, Ticket_length, Famsize] and Survived are less then 0.1
# * Apply the Pearson Residual Test to have a better insight
# -----
# #### Pearson's Residual Test
# * For a given factor, The null hypothesis of a feature is that 'The prediction of Survived will not consider this factor'.
# * Use the scipy.stats library to calculate the p-value and compare it with alpha=0.05. If the p-value between the factor and the response is larger than alpha, then this factor does not have a significant level of 95%. The null hypothesis will not be rejected. However, if the p-value is less than alpha, the factor will be rejected.

# In[ ]:


# scipy.stats to find the p-value and of each factor
# compare p-value with alpha=0.05 to find out the significance level
# select the columns for test

features= ['PassengerId', 'Pclass','Age','SibSp','Parch','Fare','Embarked','new_Sex','new_Embarked','new_Title','Ticket_length','Famsize']
# drop the missing value row to have a accurate estimation?
for feature in features:
    table = pd.crosstab(copy1[feature], copy1['Survived'], margins=False)
    stat, p, dof, expected = stats.chi2_contingency(table)
    print("The p-value of", feature,"is: ",p)
    
    


# * The p-value of PassengerId is larger than 0.05.
# * PassengerId does not have a siginificant level of 95% and can be dropped
# * Also the object types feature [Name,Sex,Ticket, Embarked,Title] should be dropped
# * Famsize is the sum of SibSp and Parch. They have high relevancy.
# * Obviously, Famsize has a much higher siginificant level than SibSp and Parch, so SibSp and Parch should be abandoned

# In[ ]:


copy1.info()


# In[ ]:




copy1 = copy1.drop(['PassengerId','Name','Sex','Embarked','Ticket','Title'],axis=1)


# In[ ]:


copy2=copy2.drop(['Name','Sex','Embarked','Ticket','Title'],axis=1)


# In[ ]:


def age(i):
    if i<=12: return 0
    elif i<=26: return 1
    elif i<=40: return 2
    elif i<=55: return 3
    else : return 4
copy1['Age']=[age(i) for i in copy1.Age]
def alone(i):
    if i > 0:return 1
    else: return 0
copy1['isAlone']=[alone(i) for i in copy1.Famsize]

def fare(i):
    # mostly class 3 and some of them are class 2
    if i<=13: return 1
    # mostly class 2 and some of them are class 1
    elif i<=26: return 2
    # mostly class 1 and some of them are class 2
    elif i<=74: return 3
    # else absolutely class 1
    else: return 4
copy1['Fare']=[fare(i) for i in copy1.Fare]

def com(i):
    if i.new_Sex==0 and i.Pclass==1: return 0
    if i.new_Sex==0 and i.Pclass==2: return 1
    if i.new_Sex==0 and i.Pclass==3: return 2
    if i.new_Sex==1 and i.Pclass==1: return 3
    if i.new_Sex==1 and i.Pclass==2: return 4
    if i.new_Sex==1 and i.Pclass==3: return 5
copy1['SP']=copy1.apply(com,axis='columns')
copy1['SF']=copy1['new_Sex']*copy1['Fare']

def com2(i):
    if i.new_Sex==0 and i.isAlone==0: return 3
    if i.new_Sex==0 and i.isAlone==1: return 2
    if i.new_Sex==1 and i.isAlone==0: return 1
    if i.new_Sex==1 and i.isAlone==1: return 0
    
copy1['sex_alone']=copy1.apply(com2,axis='columns')


copy2['Age']=[age(i) for i in copy2.Age]
copy2['isAlone']=[alone(i) for i in copy2.Famsize]
copy2['Fare']=[fare(i) for i in copy2.Fare]
copy2['SP']=copy2.apply(com,axis='columns')
copy2['SF']=copy2['new_Sex']*copy2['Fare']
copy2['sex_alone']=copy2.apply(com2,axis='columns')


copy1=copy1.drop(['SibSp','Parch','Famsize','Age'],axis=1)
copy2=copy2.drop(['SibSp','Parch','Famsize','Age'],axis=1)

copy1=copy1.drop(['Ticket_length'],axis=1)
copy2=copy2.drop(['Ticket_length'],axis=1)


# In[ ]:


# the correlation matrix
corr=copy1.corr() 
# mask the upper triangle
sns.set(style="white")
plt.figure(figsize=(11,7))
mask=np.triu(np.ones_like(corr,dtype=np.bool))
# colour
cmap=sns.diverging_palette(240,10,n=9)
# annot to display the value
sns.heatmap(corr,annot=True,mask=mask,cmap='jet',linewidths=0.6)


# * Age variable can be dropped as the correlation value is small

# * However,some of the machine learning model will treat the categorical features as numeric features, such as XGBoost

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(copy1[['new_Embarked','new_Title','SP','SF','sex_alone']]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(copy2[['new_Embarked','new_Title','SP','SF','sex_alone']]))

# One-hot encoding removed index; put it back
OH_cols_train.index = copy1.index
OH_cols_valid.index = copy2.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = copy1.drop(['new_Embarked','new_Title','SP','SF','sex_alone'], axis=1)
num_X_valid = copy2.drop(['new_Embarked','new_Title','SP','SF','sex_alone'], axis=1)

# Add one-hot encoded columns to numerical features
copy1 = pd.concat([num_X_train, OH_cols_train], axis=1)
copy2 = pd.concat([num_X_valid, OH_cols_valid], axis=1)


# -----
# <a id="11"></a> 
# # 4.Modelling
# Apply several prediction model and find the best of them

# <a id="12"></a> 
# ## 4.1 Building Machine Learning Models
# * Divide the trainging dataset into 2 groups 
# * One group for training and Another group for model testing say valid dataset

# In[ ]:



y=copy1['Survived']
copy3=copy1.drop(['Survived'],axis=1)
X_train,X_valid,y_train,y_valid = train_test_split(copy3,y,train_size=0.8,test_size=0.2,random_state=0)

mae=[]
accuracy=[]
prediction=[]


# ### Logistic Regression

# In[ ]:


l1=LogisticRegression(max_iter=150)
l1.fit(X_train,y_train)
y_pred=l1.predict(X_valid)
prediction.append(y_pred)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))


# ### Random Forest Classfier
# * Find the best number of n_estimators

# In[ ]:


min=100
a=0
y=0
l2=RandomForestClassifier(n_estimators=40,random_state=1)
print('The mean absolute error of')
for i in [60,80,100,120,140,160,180]:
    l=RandomForestClassifier(n_estimators=i,random_state=1)
    l.fit(X_train,y_train)
    y_pred=l.predict(X_valid)
    if mean_absolute_error(y_valid, y_pred)<min:
        min= mean_absolute_error(y_valid, y_pred)
        a=accuracy_score(y_valid, y_pred)
        y=y_pred
        l2=l
    print('n_estimators =',i,': ', mean_absolute_error(y_valid, y_pred))
prediction.append(y)    
mae.append(min)
accuracy.append(a)


# * We will then choose n_estimators=100

# ### K Nearest  Neighbor Classifier
# * Find the best hyperparameter by RandomizedSearchCV
# * Implement Cross-validation with cv=7 to train the model

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

y=copy1['Survived']

knn_model = KNeighborsClassifier()

params = {
    "n_neighbors": randint(5, 30),
    "p":randint(1, 4),
    "leaf_size":randint(20, 100),
    "weights":['uniform','distance']
    
}

search = RandomizedSearchCV(knn_model, param_distributions=params, random_state=42, n_iter=200, cv=7, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_train, y_train)

best_scores(search.cv_results_, 1)

# After finding the best parameter
# Build the model to predict X_valid and check its performance
l3=KNeighborsClassifier(leaf_size=87,n_neighbors=10,p=1,weights='distance')
l3.fit(X_train,y_train)
y_pred=l3.predict(X_valid)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))
prediction.append(y_pred)


# In[ ]:



"""min=100
l3=KNeighborsClassifier()
print('The mean absolute error of')
for i in [5,10,15,20,25,30]:
    l=KNeighborsClassifier(n_neighbors=i)
    l.fit(X_train,y_train)
    y_pred=l.predict(X_valid)
    if mean_absolute_error(y_valid, y_pred)<min:
        min=mean_absolute_error(y_valid, y_pred)
        a=accuracy_score(y_valid, y_pred)
        y=y_pred
        l3=l
    print('n_neighbors =',i,': ',mean_absolute_error(y_valid, y_pred))

prediction.append(y)
mae.append(min)
accuracy.append(a)"""


# ### Naive Bayes

# In[ ]:




NB=[GaussianNB(),BernoulliNB(),CategoricalNB(alpha=2),MultinomialNB()]
name=['Gaussian NB','Bernoulli NB','Categorical NB','Multinomial NB']
print('The mean absolute error of')
min=100
l4=NB[0]
for i in range(len(NB)):
    model=NB[i]
    model.fit(X_train,y_train)
    y_pred=model.predict(X_valid)
    if mean_absolute_error(y_valid, y_pred)<min:
        l4=model
        a=accuracy_score(y_valid, y_pred)
        y=y_pred
        min=mean_absolute_error(y_valid, y_pred)
    print(name[i],': ',mean_absolute_error(y_valid, y_pred))
prediction.append(y)
mae.append(min)
accuracy.append(a)


# * Categorical Naive Bayes has the best performance

# ### Stochastic Gradient Descent Classifier
# * It's equivilant to linear SVM

# In[ ]:


l5 = SGDClassifier(alpha=0.0001,epsilon=0.06, loss='log',penalty="l2", max_iter=3,random_state=0)
l5.fit(X_train, y_train)
y_pred=l5.predict(X_valid)
prediction.append(y_pred)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))
#print(mean_absolute_error(y_valid, y_pred))


# ### XGBClassifier

# * Find the best XGBM model by hypeparameter method
# * Also apply the cross validation with cv=7

# In[ ]:



def best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

y=copy1['Survived']

xgb_model = XGBClassifier(objective='binary:logistic',criterion='mae',eval_metric="auc")

params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 10), 
    "n_estimators": randint(100, 150),
    "subsample": uniform(0.6, 0.4),
    "eta":uniform(0.01, 0.3),
    "max_leaf_nodes":randint(30, 500),
    "colsample_bytree":uniform(0,1)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=7, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_train, y_train)

best_scores(search.cv_results_, 1)

l6=XGBClassifier(colsample_bytree=0.157,eta=0.058,gamma=0.167,learning_rate=0.1016,max_depth=6,max_leaf_nodes=199,n_estimators=135,subsample=0.847,objective='binary:logistic',criterion='mae',eval_metric="auc")
l6.fit(X_train,y_train)
y_pred=l6.predict(X_valid)

prediction.append(y_pred)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))


# In[ ]:


"""l6=XGBClassifier()
min=100
print('The mean absolute error of')
for i in [1,2,3,4,5,6]:
    l=XGBClassifier(max_depth=i)
    l.fit(X_train,y_train)
    y_pred=l.predict(X_valid)
    if mean_absolute_error(y_valid, y_pred)<min:
        min=mean_absolute_error(y_valid, y_pred)
        a=accuracy_score(y_valid, y_pred)
        y=y_pred
        l6=l
    print('max_depth=',i,': ',mean_absolute_error(y_valid, y_pred) )
prediction.append(y_pred)
mae.append(min)
accuracy.append(a)"""
    
  


# In[ ]:



"""from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

y=copy1['Survived']
# evaluate the model
l7 = LGBMClassifier()
#n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=1)
l7.fit(X_train,y_train)
y_pred=l7.predict(X_valid)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))"""


# In[ ]:


from lightgbm import LGBMClassifier
def best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
y=copy1['Survived']

lgb_model = LGBMClassifier()

params = {
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6), 
    "n_estimators": randint(100, 150),
    "subsample": uniform(0.6, 0.4),
    "num_leaves":randint(20,40),
    "n_splits":randint(5,30)
}

search = RandomizedSearchCV(lgb_model, param_distributions=params, random_state=42, n_iter=200, cv=7, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_train, y_train)

best_scores(search.cv_results_, 1)
l7= LGBMClassifier(learning_rate=0.116,max_depth=2,n_estimators=110,n_splits=8,num_leaves=32,subsample=0.813)
l7.fit(X_train,y_train)
y_pred=l7.predict(X_valid)

prediction.append(y_pred)
mae.append(mean_absolute_error(y_valid, y_pred))
accuracy.append(accuracy_score(y_valid, y_pred))


# <a id="13"></a> 
# ## 4.2 Model Comparison
# * Mean absolute error and accuracy score of each model

# In[ ]:



s1=pd.Series(mae,index=['Logistic_Regression', 'Random_Forest_Classifier','K_Nearest_Neighbors_Classifier','Categorical_NB','SGD_Classifier','XGB_Classifier','LGBMclassfier'])
s2=pd.Series(accuracy,index=['Logistic_Regression', 'Random_Forest_Classifier','K_Nearest_Neighbors_Classifier','Categorical_NB','SGD_Classifier','XGB_Classifier','LGBMclassfier'])
pd.concat([s1,s2],axis=1,keys=['Mean Absolute Error','Accuracy Score'])


# * XGB_Classfier has the smallest mean absolute error and the largest accuracy score

# ### Roc Curve
# * Let the sample size be: a+b+c+d
# * Let Y be the real value and $\hat{Y}$ is the prediction value

# In[ ]:


s1=pd.Series(['a','c'], index=['Y=1','Y=0'])
s2=pd.Series(['b','d'], index=['Y=1','Y=0'])
pd.concat([s1,s2],axis=1,keys=['$\hat{Y}$=1', '$\hat{Y}$=0'])


# Then we can calculate sensitivity and Specificity:
# * **Sensitivity**: $P(\hat{Y}=1|Y=1) = \frac{a}{a+b}$
# * **Specificity**: $P(\hat{Y}=0|Y=0) = \frac{d}{c+d}$
# * The ROC curve is plotting **sensitivity** in y-axis and **1-specificity** in x-axis. 
# * AUC: Concordance index c is the area under ROC curve. The bigger the c, the better the model.

# In[ ]:


sns.set(style="darkgrid")
fpr = dict()
tpr = dict()
thresholds=dict()
roc_auc = dict()

fpr[0], tpr[0], thresholds[0] = roc_curve(y_valid, l1.predict_proba(X_valid)[:,1])
fpr[1], tpr[1], thresholds[1] = roc_curve(y_valid, l2.predict_proba(X_valid)[:,1])
fpr[2], tpr[2], thresholds[2] = roc_curve(y_valid, l3.predict_proba(X_valid)[:,1])
fpr[3], tpr[3], thresholds[3] = roc_curve(y_valid, l4.predict_proba(X_valid)[:,1])
fpr[4], tpr[4], thresholds[4] = roc_curve(y_valid, l5.predict_proba(X_valid)[:,1])
fpr[5], tpr[5], thresholds[5] = roc_curve(y_valid, l6.predict_proba(X_valid)[:,1])
fpr[6], tpr[6], thresholds[6] = roc_curve(y_valid, l7.predict_proba(X_valid)[:,1])
#roc_auc[0] = auc(fpr[0], tpr[0])
for i in range(7):
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,10))
colors = ['aqua', 'darkorange', 'cornflowerblue','deeppink','chartreuse','yellow','green']
plt.plot(fpr[0], tpr[0], label='Logistic Regression (area = %0.2f)' % roc_auc[0],color=colors[0])
plt.plot(fpr[1], tpr[1], label='Random Forest Classifier (area = %0.2f)' % roc_auc[1],color=colors[1])
plt.plot(fpr[2], tpr[2], label='K Nearest Neighbors Classifier (area = %0.2f)' % roc_auc[2],color=colors[2])
plt.plot(fpr[3], tpr[3], label='Categorical NB (area = %0.2f)' % roc_auc[3],color=colors[3])
plt.plot(fpr[4], tpr[4], label='SGD Classifier (area = %0.2f)' % roc_auc[4],color=colors[4])
plt.plot(fpr[5], tpr[5], label='XGB Classifier (area = %0.2f)' % roc_auc[5],color=colors[5])
plt.plot(fpr[6], tpr[6], label='LGBM Classifier (area = %0.2f)' % roc_auc[6],color=colors[6])
plt.plot([0, 1], [0, 1], linestyle='--',label='Base',color='black')
plt.legend()
plt.show()


# * XGB Classifier, Random Forest Classifier and K Nearest Neighbors Classifier have the largest AUC value that are the best performance among all the models
# * Also KNN Classifier have the smallest mean absolute error and the largest accuracy score. We gonna take the average result of these 3 models
# 

# ----
# <a id="14"></a> 
# # 5.Data Submisson

# * The format of submission:

# In[ ]:


file_path3 = "../input/titanic/gender_submission.csv"
sample = pd.read_csv(file_path3)
sample.info()


# * Prediction on test dataset
# * What we gonna do is take the mean value of KNN, LGBM and XGB models' prediction

# In[ ]:



s1=copy2['PassengerId']
copy2=copy2.drop(['PassengerId'],axis=1)
knn_pred=l3.predict(copy2)
xgb_pred=l6.predict(copy2)
rf_pred=l2.predict(copy2)
sum_pred=knn_pred+xgb_pred+rf_pred

def mean(i):
    if i/3>0.5: return 1
    else: return 0
copy2['Survived']=[mean(i) for i in sum_pred]
s2=pd.concat([s1,copy2['Survived']],axis=1)


# In[ ]:


submission=pd.concat([s1,copy2['Survived']],axis=1)
submission.to_csv("titanic_submission8.csv", index=False)

