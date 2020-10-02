#!/usr/bin/env python
# coding: utf-8

# # About the data

# # Contaxt
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
# this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 
# # Attributes
# 1) age
# 2) sex
# 3) chest pain type (4 values)
# 4) resting blood pressure
# 5) serum cholestoral in mg/dl
# 6) fasting blood sugar > 120 mg/dl
# 7) resting electrocardiographic results (values 0,1,2)
# 8) maximum heart rate achieved
# 9) exercise induced angina
# 10) oldpeak = ST depression induced by exercise relative to rest
# 11) the slope of the peak exercise ST segment
# 12) number of major vessels (0-3) colored by flourosopy
# 13) thal: 3 = normal; 6 = fixed defect; 7 = reversable defec
# 
# 

# # LOAD PACKAGES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.preprocessing import scale
from scipy import stats
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# # LOADING OF DATA

# In[ ]:


df = pd.read_csv('../input/heart-disease-uci/heart.csv')


# # READING THE DATA AND SEE BASIC FEATURES

# In[ ]:


df.shape


# this data has 303 rows and 14 columns

# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.tail()


# in the dataframe we can see that there is a column name target where values are only binary. if 0 is there than Heart Desiease not present and vice versa.

# In[ ]:


df.isna().sum()


# we can see that the no of missing value is none.

# In[ ]:


df.info()


# All the columns are numeric so we are free to do any type of numerical analysis.

# # Statistical analysis

# In[ ]:


df.describe().style.background_gradient(cmap='Reds')


# Here we can find the max value, min value, quantiles, St. deviation, mean, median for each column. darker the color higher the value and lighter the color smaller the value.

# In[ ]:


#correlation in between the columns
df.corr().style.background_gradient(cmap="Greens")

Here we can infer that target is highly corrlated to the chest pain, max heart rate and cholestorel level.
# # Normality Testing

# In[ ]:


df.skew().sort_values()


# In[ ]:


df.kurt().sort_values()


# 1. If skewness is less than -1 or greater than 1, the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed. If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# 2. Kurtosis of the normal distribution, which is equal to 3. If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution.
# 3. Here we can infer that the data set is little bit skewed but not prevelent in the outlier. So we can say that its a moderate data set.

# # OLS (Ordinary least Square Regression Model)::traget W.R.T cholestorel, chest pain and max heart rate.

# In[ ]:


y = df['target']
x =  df[['chol','cp','thalach']]
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


# In[ ]:


print('Parameters: ', results.params)
print('Standard errors: ', results.bse)


# # Visualization

# categorical features

# In[ ]:


sns.scatterplot(x = 'chol', y = 'trestbps', color = 'green', data = df)
plt.xlabel('cholestrol')            
plt.ylabel('trestbps') 
plt.title('Cholestrol Vs Trestbps');


# 

# In[ ]:


fig = px.histogram(df, y="cp", x="age", color="sex",marginal="rug",hover_data=df.columns)
fig.show()


# Visualise chet pain, accross the gender along with age. We can infer that age between 50 to 60 people have more chest pain and no of male is greater than female.

# In[ ]:


fig = px.scatter(df, x="trestbps", y="chol", color="age",size='cp', hover_data=['sex'])
fig.show()


# 1. chestpain 0,1,2,3 are 'Typical','Atypical','Non-Anginal','Aysyptomatic' respectively
# 2. sex:: 0=female, 1=male

# In[ ]:


fig = px.box(df, x="age", y="chol", points="all")
fig.show()


# age vs cholestorel plotting.

# In[ ]:


#thal 3 = normal, 6 = fixed defect, 7 = reversable defect (category feature)
fig, ax=plt.subplots(1,2,figsize=(20,8))
sns.countplot(x='thal',data=df,hue='target',palette='Set2',ax=ax[0])
ax[0].set_xlabel("number of major vessels colored by flourosopy")
df.thal.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Reds');


# In[ ]:


sns.jointplot(x = 'cp', y = 'age', kind = 'kde', color ='Green', data = df);


# In the above joint plot we can infer that for heart disease chest pain is a common symptom and it is a proprotional to age. higer the age, denser the chest pain. 

# In[ ]:


df1=df[['age','restecg']]


# In[ ]:


fig= go.Figure(go.Funnelarea(text=df1['age'],values=df1['restecg']))
fig.show()


# Here age wise ecg rate is visualise. for more details please hover on the plot.

# # Continuous Variables
# 

# In[ ]:


df['heartratemax']=240-df['age']
df['heartrateratio']=df['thalach']/df['heartratemax']


# In[ ]:


df2=df[['age','trestbps','chol','thalach','oldpeak','ca','heartratemax','heartrateratio','target']]


# In[ ]:


sns.pairplot(df2)


# In the pair plot we are tring to visualise the pairwise relation inbetween the continous variables.

# continous assesment with chol.

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(20,8))
sns.boxenplot(y='chol',data=df,x='sex',hue='target',palette='twilight',ax=ax[0,0])
ax[0,0].set_title("Cholestrol V/S Sex");
sns.boxenplot(y='chol',data=df,x='cp',hue='target',ax=ax[0,1],palette='Spectral')
ax[0,1].set_title("Cholestrol V/S Chest Pain");
sns.swarmplot(y='chol',data=df,x='thal',hue='target',ax=ax[1,0],palette='copper')
ax[1,0].set_title("Cholestrol V/S Thalium stress test result")
sns.swarmplot(y='chol',data=df,x='oldpeak',hue='target',ax=ax[1,1],palette='Set2')
ax[1,1].set_title("Cholestrol V/S ST depression induced by exercise relative to rest");
plt.xticks(rotation=90)
plt.grid()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# On the above plot we can infer that for diseased Cholestrol level is directly proprotional.

# Variable Differences by Target Value

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
sns.violinplot(x="target", y="age", data=df,color = 'pink',ax=axes[0][0]).set_title('Age')
sns.swarmplot(x="target", y="age", data=df,ax = axes[0][0])

sns.violinplot(x="target", y="trestbps", data=df,color = 'pink',ax = axes[0][1]).set_title('Resting Blood Pressure')
sns.swarmplot(x="target", y="trestbps", data=df,ax = axes[0][1])

sns.violinplot(x="target", y="chol", data=df,color = 'pink',ax = axes[1][0]).set_title('Cholesterol')
sns.swarmplot(x="target", y="chol", data=df,ax = axes[1][0])

sns.violinplot(x="target", y="thalach", data=df,color = 'pink',ax = axes[1][1]).set_title('Max Heart Rate Achieved')
sns.swarmplot(x="target", y="thalach", data=df,ax = axes[1][1])

sns.violinplot(x="target", y="oldpeak", data=df,color = 'pink',ax = axes[2][0]).set_title('ST Depression Peak')
sns.swarmplot(x="target", y="oldpeak", data=df,ax = axes[2][0])

sns.violinplot(x="target", y="heartrateratio", data=df2,color = 'pink',ax = axes[2][1]).set_title('Peak Heart Rate to Max Heart Rate Ratio')
sns.swarmplot(x="target", y="heartrateratio", data=df2,ax = axes[2][1]);

plt.grid()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# # correlation matrix between different variables

# In[ ]:


plt.figure(figsize=(30,10))
sns.heatmap(df.corr(), annot=True);


# There are no features with a pretty strong correlation (above |0.7|)

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 8))
women = df[df['sex'] == 0]
men = df[df['sex'] == 1]

ax = sns.distplot(women[women['target'] == 1].age, bins=18, label = 'sick', ax = axes[0], kde =False, color="green")
ax = sns.distplot(women[women['target'] == 0].age, bins=40, label = 'not_sick', ax = axes[0], kde =False, color="red")
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['target']==1].age, bins=18, label = 'sick', ax = axes[1], kde = False, color="green")
ax = sns.distplot(men[men['target']==0].age, bins=40, label = 'not_sick', ax = axes[1], kde = False, color="red")
ax.legend()
ax.set_title('Male');


# On the basis of gender and age visualise the diseased and healthy people

# In[ ]:


df["sex"].value_counts()


# In[ ]:


fig,ax=plt.subplots(1, 3, figsize=(20, 8))
sns.countplot(x = "sex", hue = "target", data = df, ax = ax[0])
sns.swarmplot(x = "sex", y = "age", hue = "target", data = df, ax = ax[1])
sns.violinplot(x = "sex", y = "age", hue= "target", split = True, data = df, ax=ax[2])
sns.despine(left=True)
plt.legend(loc="upper right");


# On the basis of sex, differnt plots are ploted, where we can infer taht male people are highly diseased compared with female. 

# # thal 3 = normal; 6 = fixed defect; 7 = reversable defect 

# In[ ]:


df.thal.unique()
sns.boxenplot(x = "thal", y = "trestbps", data = df, hue="target")


# Visualise the diseased and helathy people W.R.T blood pressure and thal lavel**

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,10))
var3 = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for idx, feature in enumerate(var3):
    ax = axes[int(idx/4), idx%4]
    if feature != 'target':
        sns.countplot(x=feature, hue='target', data=df, ax=ax)


# Chest pain: the heart desease diagnosis is greater among the patients that feel any chest pain.
# 
# Restegc - Eletrocardiagraph results: the rate of heart desease diagnoses higher for patients with a ST-T wabe abnormality .
# 
# Slope: The ratio of patients diagnosed with heart desease is higher for slope = 2
# 
# Ca: The diagonosed ratio decreases fo ca between 1 and 3.
# 
# Thal: the diagnosed ratio is higher for thal = 2.

# # if you like my analysis don't forget to give an upvote.
# # Feel free to give any suggestion :) 

# In[ ]:




