#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
import scipy.stats as ss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/travel insurance.csv")
df1=df
df.head(5)


# In[ ]:


df.info()


# So, there are 4 numerical columns and 7 categorical columns

# Now, lets check which columns have the null values

# In[ ]:


missingno.matrix(df)


# "Black" in the data depicts the column is fill with data and "White" depicts they have null values in that particular area
# 
# So, we can conclude that only "Gender" have the null values and seems quite much

# Lets see whats the number of null values in the "Gender" column

# In[ ]:


df['Gender'].isnull().sum()


# 45107/63326 are null values, nearly 71.2% data in the column are null values.

# I have replaced the null values with another category called "Not Specified"

# In[ ]:


df.fillna('Not Specified',inplace=True)


# In[ ]:


df.isnull().sum()


# There is no null values now

# First make an another dataframe which consist of only numerical columns from df

# In[ ]:


df_numerical=df._get_numeric_data()
df_numerical.info()


# ****Now lets look at the spread of the numerical data****

# In[ ]:


for i, col in enumerate(df_numerical.columns):
    plt.figure(i)
    sns.distplot(df_numerical[col])


# **From the graph we can conclude:**
# 
# *Duration*: Data in this column is highly right skewed.
# 
# *Net Sales and Commison*: These both column seems to related but the graph plot shows disparency as low net sales     shows high commison which is not pratically possible.
# 
# *Age*: Age is random so its distribution can be random.

# **Lets check the data in "Duration" column**

# In[ ]:


df['Duration'].describe()


# We have negative values in this Duration column but can time be negative? **NO**

# **Lets see how many negative values we have in Duration column**

# In[ ]:


df10=df['Duration']<0
df10.sum()


# So, there are 5 negative values in Duration column. I am gonna replace those with the mean value

# In[ ]:


df.loc[df['Duration'] < 0, 'Duration'] = 49.317


# Previously we have checked that some columns have low Net Sales but High Commison but thats not possible
# 
# Lets see how many such columns we have here

# In[ ]:


df6= df['Net Sales']<df['Commision (in value)']
df6.sum()


# We gonna make all comission value 0 where net sales is 0.

# In[ ]:


df.loc[df['Net Sales'] == 0.0, 'Commision (in value)'] = 0


# In[ ]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[ ]:


categorical=['Agency', 'Agency Type', 'Distribution Channel', 'Product Name',  'Destination','Gender','Claim']
cramers=pd.DataFrame({i:[cramers_v(df[i],df[j]) for j in categorical] for i in categorical})
cramers['column']=[i for i in categorical if i not in ['memberid']]
cramers.set_index('column',inplace=True)

#categorical correlation heatmap

plt.figure(figsize=(10,7))
sns.heatmap(cramers,annot=True)
plt.show()


# **Observation:**
# 
# We can see the Co-relation between the Categorical columns.
# 
# Can coclude that the cloumn "Agency Type' have high corelation with some of the columns like "Agency","Product Name" thus we can drop "Agency Type".

# **Lets Check how Gender is related to the Claim column**

# In[ ]:


test=[(df[df['Gender']=='Not Specified']['Claim'].value_counts()/len(df[df['Gender']=='Not Specified']['Claim']))[1],(df[df['Gender']=='M']['Claim'].value_counts()/len(df[df['Gender']=='M']['Claim']))[1],
      (df[df['Gender']=='F']['Claim'].value_counts()/len(df[df['Gender']=='F']['Claim']))[1]]
test


# In[ ]:


fig, axes=plt.subplots(1,3,figsize=(24,9))
sns.countplot(df[df['Gender']=='Not Specified']['Claim'],ax=axes[0])
axes[0].set(title='Distribution of claims for null gender')
axes[0].text(x=1,y=30000,s=f'% of 1 class: {round(test[0],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')
sns.countplot(df[df['Gender']=='M']['Claim'],ax=axes[1])
axes[1].set(title='Distribution of claims for Male')
axes[1].text(x=1,y=6000,s=f'% of 1 class: {round(test[1],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')
sns.countplot(df[df['Gender']=='F']['Claim'],ax=axes[2])
axes[2].set(title='Distribution of claims for Female')
axes[2].text(x=1,y=6000,s=f'% of 1 class: {round(test[2],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')
plt.show()


# **Observation :**
# Female Gender have highest number of Claims approved. Though the numbers aren't that descriptive to conclude to a decision

# In[ ]:


pd.crosstab(df['Agency'],df['Agency Type'],margins=True)


# **Does Claim % depends on Agency Type**

# In[ ]:


table1=pd.crosstab(df['Agency'],df['Claim'],margins=True)

table1.drop(index=['All'],inplace=True)
table1=(table1.div(table1['All'],axis=0))*100

table1['mean commision']=df.groupby('Agency')['Commision (in value)'].mean()
table1


# In[ ]:


table1.columns


# In[ ]:


fig,ax1=plt.subplots(figsize=(18,9))
sns.barplot(table1.index,table1.Yes,ax=ax1)
plt.xticks(rotation=90)
ax1.set(ylabel='Acceptance %')
ax2=ax1.twinx()
sns.lineplot(table1.index,table1['mean commision'],ax=ax2,linewidth=3)


# **X axis= Agency Name
# Y axis=Acceptance %
# line plot= Commision
# So we can see certain agencies have higher % of acceptance, C2B, LWC, TTW**

# In[ ]:


table2=pd.crosstab(df['Product Name'],df['Claim'],margins=True)
table2=(table2.div(table2['All'],axis=0))*100

table2['mean commision']=df.groupby('Product Name')['Commision (in value)'].mean()
table2.drop(index=['All'],inplace=True)
table2


# In[ ]:


fig,ax1=plt.subplots(figsize=(20,11))
sns.barplot(table2.index,table2.Yes,ax=ax1)
plt.xticks(rotation=90)
ax1.set(ylabel='Acceptance %')
ax2=ax1.twinx()
sns.lineplot(table2.index,table2['mean commision'],ax=ax2,linewidth=3)


# **Some of the products with high commission have a high ratio of claims acceptance. The plans with zero commission generally have low acceptance**

# **Does the duration of the trip have an impact on claims acceptance**

# In[ ]:


tests=df.copy()
tests['Duration_label']=pd.qcut(df['Duration'],q=35)
table3=pd.crosstab(tests['Duration_label'],tests['Claim'],normalize='index')
table3


# In[ ]:


table3.columns


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(table3.index,table3.Yes)
plt.xticks(rotation=90)


# **On varying the value of bins, we found that Durations>364 have a high percentage of acceptance compared to the rest.**

# In[ ]:


table4=pd.crosstab(df['Destination'],df['Claim'],margins=True,normalize='index')
table4


# In[ ]:


table4 = table4.sort_values(by=['Yes'], ascending=[False])
table4


# In[ ]:


sns.countplot(df['Claim'])


# From the above graph we can say that there is high imbalance in the target variable. We will see how to deal with that a little later

# Lets see which features are important for the prediction using **Chi Square Test**

# In[ ]:


from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, df):
        self.df = df
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = ss.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


# In[ ]:


X = df.drop(['Claim'], axis=1)
ct = ChiSquare(df)
for c in X.columns:
    ct.TestIndependence(c, 'Claim')


# According to the **Chi Square Test** Distribution Channel is not important thus I am dropping the column. Also as discussed above "Agency Type" is gonna dropped

# In[ ]:


df.drop(columns=['Distribution Channel','Agency Type'],axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


y=df['Claim']


# In[ ]:


x=df
x.drop(columns='Claim',axis=1,inplace=True)


# In[ ]:


x_dummy=pd.get_dummies(x,columns=['Agency','Gender','Product Name','Destination'],drop_first=True)


# In[ ]:


lr = LogisticRegression()
rfe = RFE(estimator=lr, n_features_to_select=10, verbose=3)
rfe.fit(x_dummy, y)
rfe_df1 = rfe.fit_transform(x_dummy, y)


# In[ ]:


print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x_dummy.columns)))


# In[ ]:


X=x_dummy[['Agency_EPX','Agency_TST','Gender_Not Specified','Product Name_2 way Comprehensive Plan','Product Name_24 Protect','Product Name_Basic Plan','Product Name_Comprehensive Plan','Product Name_Premier Plan','Product Name_Travel Cruise Protect','Product Name_Value Plan']]


# In[ ]:


X.head(5)


# In[ ]:




smote = SMOTE(random_state=7)
X_ov, y_ov = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_ov, y_ov, train_size=0.7, random_state=7)


# In[ ]:


from sklearn.svm import LinearSVC
algo_dict = {'Random Forest Classifier':RandomForestClassifier(),'DecisionTreeClassifier':DecisionTreeClassifier(),'Linear SVC':LinearSVC()}


                
algo_name=[]
for i in algo_dict:
    algo_name.append(i)

for i in algo_dict.keys():
      
          
        algo = algo_dict[i]
        model = algo.fit(X_train, y_train)
        y_pred = model.predict(X_test)        
        print('Classification report'+'\n',classification_report(y_test, y_pred))
        print('***'*30)
          
        


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    print()


# In[ ]:


gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

#print("Confusion Matrix:")
#print(confusion_matrix(y_test, predictions))
#print()
print("Classification Report")
print(classification_report(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




