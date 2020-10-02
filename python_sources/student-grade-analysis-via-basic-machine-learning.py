#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder

warnings.filterwarnings("ignore")


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing and checking data

# In[ ]:


#importing the data
df = pd.read_csv("/kaggle/input/student-grade-prediction/student-mat.csv")
df.info()


# In[ ]:


#creating a correlation matrix to understand the correlation between the features
df.corr()


# In[ ]:


#using seaborn heatmap to analyse the correlation graphically
import seaborn as sns
sns.heatmap(df.corr(),xticklabels=df.corr().columns,yticklabels=df.corr().columns)


# G3, apart from G1 and G2, is most correlated with Mother's Education(Medu), Father's Education(Fedu), and studytime.

# In[ ]:


#checking 
df.describe()


# In[ ]:


#checking data types of column values and presence of null values(if any)
df.info()


# There are 33 columns in this data. These columns signify the following:
# 
# **school** - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# **sex** - student's sex (binary: 'F' - female or 'M' - male)
# **age** - student's age (numeric: from 15 to 22)
# **address** - student's home address type (binary: 'U' - urban or 'R' - rural)
# **famsize** - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# **Pstatus** - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# **Medu** - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# **Fedu** - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# **Mjob** - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# **Fjob** - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# **reason** - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# **guardian** - student's guardian (nominal: 'mother', 'father' or 'other')
# **traveltime** - home to school travel time (numeric: 1 - 1 hour)
# **studytime** - weekly study time (numeric: 1 - 10 hours)
# **failures** - number of past class failures (numeric: n if 1<=n<3, else 4)
# **schoolsup** - extra educational support (binary: yes or no)
# **famsup** - family educational support (binary: yes or no)
# **paid** - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# **activities** - extra-curricular activities (binary: yes or no)
# **nursery** - attended nursery school (binary: yes or no)
# **higher** - wants to take higher education (binary: yes or no)
# **internet** - Internet access at home (binary: yes or no)
# **romantic** - with a romantic relationship (binary: yes or no)
# **famrel** - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# **freetime** - free time after school (numeric: from 1 - very low to 5 - very high)
# **goout** - going out with friends (numeric: from 1 - very low to 5 - very high)
# **Dalc** - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# **Walc** - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# **health** - current health status (numeric: from 1 - very bad to 5 - very good)
# **absences** - number of school absences (numeric: from 0 to 93)
# 
# These grades are related with the course subject, Math or Portuguese:
# 
# **G1** - first period grade (numeric: from 0 to 20)
# **G2** - second period grade (numeric: from 0 to 20)
# **G3** - final grade (numeric: from 0 to 20, output target)

# ### Exploratory Data Analysis

# In[ ]:


#distribution of female to male in the given data
total_count=df['sex'].value_counts()
plt.pie(x=total_count,colors=['lightskyblue','red'], labels=['Male','Female'],autopct='%1.1f%%',shadow=True)
plt.show()


# In[ ]:


#plot on traveltime vs G3 based on sex 
g=sns.boxplot(x='traveltime',y='G3',data=df,hue='sex')
g.set(xticklabels=['near','moderate','far','very far'])
plt.show()


# Males with smaller traveltime scored better than those who required more time to travel. The distribution in case of females is less as the travel time increases.

# In[ ]:


#Father's education vs final score
sns.boxplot(x='Fedu',y='G3',data=df)


# Interesting plot here,as those students score higher whose their Father's education is lower, when compared to those whose Fathers have more education

# In[ ]:


#countplot of age based on sex
sns.countplot('age',hue='sex',data=df)


# In[ ]:


#failures vs final score
g=sns.swarmplot(x='failures',y='G3',data=df)
g.set(xticklabels=['very low','low','moderate','high','very high'])
plt.show()


# In[ ]:


#freetime vs final Score
g=sns.boxplot(x='freetime',y='G3',data=df)
g.set(xticklabels=['very low','low','moderate','high','very high'])
plt.show()


# Contrary to popular believe, students with more free time have better score than those with less free time.

# In[ ]:


#outing vs final score
g=sns.swarmplot(x='goout',y='G3',data=df)
g.set(xticklabels=['very low','low','moderate','high','very high'])
plt.show()


# Most of the students, according to the plot, go out frequently,and still manage to score good marks(above average)

# In[ ]:


sns.barplot(x='age',y='absences',data=df)


# The absences increases steadily till 19, but reduce by 20 and 21. By the time they are 22, they lose all hope and dont bother
# coming alltogether

# In[ ]:


#mother's job plot
sns.countplot('Mjob',data=df)


# In[ ]:


#plot of age vs final score based on pursuing higher education
sns.barplot(x='age',y='G3',data=df,hue='higher')


# In[ ]:


#health vs final score barplot
g=sns.boxplot(x='health',y='G3',data=df)
g.set(xticklabels=['worst','low','moderate','good','excellent'])
plt.show()


# Interestingly, the students with worst health score more than those with better health status

# In[ ]:


#countplot of age vs final score based on being paid
sns.barplot(x='age',y='G3',data=df,hue='paid')


# In[ ]:


#boxplot of studytime vs final score based on internet usage
g=sns.boxplot(x='studytime',y='G3',hue='internet',data=df)
g.set(xticklabels=['very low','low','high','very high'])
plt.show()


# Students who study more and habve internet usage have better final score, as expected

# In[ ]:


g=sns.boxplot(x='famrel',y='Walc',data=df)
g.set(xticklabels=['very low','low','moderate','high','very high'])
plt.show


# In[ ]:


#age vs final score based on romantic life
sns.barplot(x='age',y='G3',hue='romantic',data=df)


# Students with romantic relationships score less than those without, with age of 20 being an exception. The students mature by 20
# and handle both the things well.

# In[ ]:


g=sns.countplot(x='goout',hue='romantic',data=df)
g.set(xticklabels=['very low','low','moderate','high','very high'])
plt.show()


# ### Model Preparation

# Most of the students who go out are very likely to be in a romantic relationship

# In[ ]:


#extracting major features only
df_features=df[['G1','G2','Medu','Fedu','studytime']]
df_features.head()
df_label=df[['G3']]


# In[ ]:


#getting values as numpy arrays for splitting
X=df_features.values
y=df_label.values


# In[ ]:


#splitting the X and y values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


#performing the regression on various models and storing the scores
scores={}
def classifier():
    dict_models={
        'Linear Regression':LinearRegression(),
        'Support Vector Machine':SVR(kernel='linear',degree=1),
        'Decision Tree':DecisionTreeRegressor(criterion='mae'),
        'Random Forest':RandomForestRegressor(n_estimators=150,criterion='mse',verbose=0)
    }
    X_train.shape
    y_train.shape
    
    for key,value in dict_models.items():
        regression=value.fit(X_train,y_train)
        score=cross_val_score(regression,X,y,scoring='neg_mean_squared_error')
        score=np.sqrt(-score.mean())
        scores[key]=score
        print(
            f'Model Name: {key},RMSE score: {(score.mean())}')


# In[ ]:


classifier()


# In[ ]:


#scaling the values(although it doesn't change the rmse)
from sklearn.preprocessing import MinMaxScaler
sc_s=MinMaxScaler()
X_train=sc_s.fit_transform(X_train)
X_test=sc_s.transform(X_test)


# In[ ]:


classifier()


# In[ ]:


#labelling the categorical column values of the dataframe
for column in df.columns:
    if df[column].dtype=='object':
        df[column]=LabelEncoder().fit_transform(df[column])


# In[ ]:


#extracting all the features this time for evaluation
#Only Random Forest and Decision Tree are used because others require one-hot-encoding, which we will cover
#in future notebooks
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


#Using Random Forest Regressor
new=RandomForestRegressor()
model=new.fit(X_train,y_train)
score=cross_val_score(model,X,y,scoring='neg_mean_squared_error')
score=np.sqrt(-score.mean())
scores['Random Forest Labled']=score


# In[ ]:


#Using Decision Tree Regressor
test=DecisionTreeRegressor()
model=new.fit(X_train,y_train)
score=cross_val_score(model,X,y,scoring='neg_mean_squared_error')
score=np.sqrt(-score.mean())
scores['Decision Tree Labled']=score


# ### Visualing and comparing Models

# In[ ]:


#Converting scores to datafram
scores=(pd.Series(scores)).to_frame()


# In[ ]:


#renaming the column names
scores=scores.rename(columns={0:'RMSE Error'})
scores


# In[ ]:


#plotting the scores of each model for better comparison
scores.plot(kind='bar')

