#!/usr/bin/env python
# coding: utf-8

# # import all librabries and packages

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
from sklearn.preprocessing import scale
from scipy import stats
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')


# # load data

# In[ ]:


df=pd.read_csv('../input/cancer-data-2017/cancer2017.csv', engine='python')


# # now basic overview of the data set.

# In[ ]:


df.shape


# this data set has 51 rows and 11 columns

# In[ ]:


#viewing columns...
df.columns


# In[ ]:


#redefine the column name for smooth analysis
df.columns = [c.strip() for c in df.columns.values.tolist()]
df.columns = [c.replace(' ','') for c in df.columns.values.tolist()] 
df.columns


# In[ ]:



df.rename(columns = {'Brain/nervoussystem':'Brain', 'Lung&bronchus':'Lung','Non-HodgkinLymphoma':'Lymphoma','Colon&rectum':'Colon'}, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# there are some non ascii value present so fill that value with NAN

# In[ ]:


df.replace({r'[^\x00-\x7F]+':np.nan}, regex=True, inplace=True)
df.head()


# In[ ]:


df.info()


# Except state all the other columns are consist of numbers but due to the presence of comma dtype shown as object now clearing the comma and convert the data type to numeric.

# In[ ]:


for i in range(0,df.shape[0]): 
    for j in range(1,df.shape[1]): 
        if ',' in str(df.iloc[i][j]): 
            df.iloc[i][j]=df.iloc[i][j].replace(',','') 
df.head()


# In[ ]:


df=df.apply(pd.to_numeric, errors='ignore')
df.info()


# In[ ]:


df1=df.ffill(axis=0)
df1.head(10).style.background_gradient(cmap='Reds')


# In[ ]:


stats=df1.describe().style.background_gradient(cmap='icefire')
stats


# #1) From the above table we can find the mean, median, max value, min value, sd and quantile in a graphical manner the pink color shows the max values, otherwise darkr the color greater the value in blue, the condition is [ pink > blue  ]**

# In[ ]:


df1.corr().style.background_gradient(cmap='bone')


# here dark are weakly correlated and lighter color strongly correlated.

# # normality testing

# In[ ]:


df1.skew().sort_values()


# In[ ]:


df1.kurt().sort_values()


# #1) If skewness is less than -1 or greater than 1, the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed. If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# #2) Kurtosis of the normal distribution, which is equal to 3. If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution.
# #3) Now, From the above testing we can seee that the data is not that normal or symmetric.
# #4) We can also infer that liver is the most nonsymmetric cancer type where as lung is very consistant with the normality of the data

# # visualization

# #heatmap

# In[ ]:


var=['State', 'Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver', 'Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
plt.figure(figsize=(30,10))
corr = df1[var].corr()
sns.heatmap(data=corr, annot=True);


# #1) From the haetmap we can infer that liver and lung is very weakely correlated. where as Lung & bronchus, female Brest, leukemia, overy and prostate is highly correlated.
# #2) The lighter the color correlation is strong and darker the color correlation is weak.

# lets randomly observe some OLS(Ordinary least Squre Regression)

# In[ ]:


print('correlation in betwn Liver and Pancares:',pearsonr(df1.Liver, df1.Pancreas))
print(sm.OLS(df1.Liver, df1.Pancreas).fit().summary())
chart =sns.lmplot(y= 'Liver', x='Pancreas', data=df1)


# In[ ]:


print('correlation in betwn Lung and Prostate:',pearsonr(df1.Lung, df1.Prostate))
print(sm.OLS(df1.Lung, df1.Prostate).fit().summary())
chart =sns.lmplot(y= 'Lung', x='Prostate', data=df1)


# In[ ]:


print('correlation in betwn barin and Ovary:',pearsonr(df1.Brain, df1.Ovary))
print(sm.OLS(df1.Brain, df1.Ovary).fit().summary())
chart =sns.lmplot(y= 'Brain', x='Ovary', data=df1)


# In[ ]:


print('correlation in betwn Liver and Lung & bronchus:',pearsonr(df1.Liver, df1.Lung))
print(sm.OLS(df1.Liver, df1.Lung).fit().summary())
chart =sns.lmplot(y= 'Liver', x='Lung', data=df1)


# #1) from all the above analysis we can see that log likely hood is come under-270 to -430 which not that much high as higher the log likelihood weaker the model or bad data. here we can infer that the data is not that much bad.
# #2) AIC and BIC panalise the data. lower the AIC and BIC value indicates a better fit. here from the statistic it can be infer that it's not a bad one at the same time it's not a good also.
# https://www.methodology.psu.edu/resources/AIC-vs-BIC/

# # lets see a pair wise relationship in the dataset.
# #1) As this data related to USA and in USA 'LUNG Cancer' is the second most prominent cancer so, lets see the pair wise relation of Lung cancer . (https://www.healthline.com/health/most-common-cancers)

# In[ ]:


var=['Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver','Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
sns.pairplot(df1,palette='coolwarm',hue= 'Lung')


# # state wise cancer count

# In[ ]:


var1=df1.loc[:, df.columns != 'State']
type(var1)
var1=list(var1)


# In[ ]:


type(var1)


# In[ ]:


z = df1[var1].groupby(df1['State']).sum() #plotting the state w.r.t cancer
z.T.plot(kind='barh', figsize=(20,10));


# In[ ]:


y=list(df1.columns)
x='State'
i=1
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
for row in ax:
    for col in row:
        col.bar(df1[x],df1[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.title(var1[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()


# #1) From the Both ploting we can infer that California is the biggest hotspot for any cancer, second is Florida, 3rd is Texas.

# In[ ]:


s=df1.Brain+df1.Femalebreast+df1.Colon+df1.Leukemia+df1.Liver+df1.Lung+df1.Lymphoma+df1.Ovary+df1.Pancreas+df1.Prostate


# In[ ]:


df2=df1.assign(s=s)
df3=df2[['State','s']]
df3.head(10).style.background_gradient(cmap='Oranges')


# In[ ]:


fig = go.Figure(go.Funnelarea(text =df3.State,values = df3.s))
fig.show()


# in the above one we can infer the percent of cancer in state wise. please hover the cursor on the plot. the plot is in an alphabetical order.

# # PLOT WITH RESPECT TO MEAN OF EACH ROW

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(10,5))
sns.distplot(df1[var1].mean(axis=1),bins=30,color='red');


# # PLOT WITH RESPECT TO SD OF EACH ROW

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(10,5))
sns.distplot(df1[var1].std(axis=1),bins=30,color='green');


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 8)
df1.plot(kind='bar', stacked=True);
labels = []
for r in df1.iloc[:,0]:
    labels.append(r)
plt.xticks(np.arange(50), labels, rotation=90);
plt.xlabel('Cancer');
plt.ylabel('count');


# # Boxplot of numerical variables

# In[ ]:


var=['Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver','Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
plt.figure(figsize=(20,8))
df1[var].boxplot()
plt.title("Numerical variables cancer", fontsize=20)
plt.show()


# 1) from the above box plot we can see that lung cancer is maximum, ovary is the minimum colon is the third one. except brain data, all other data is not that bad.
# 

# # Joint Distribution

# In[ ]:


#lung vs liver
plt.figure(figsize=(20,8))
plt.xlabel("lung")
plt.ylabel("liver")
plt.suptitle("Joint distribution of lung vs liver", fontsize= 15)
plt.plot(df1['Lung'], df1['Liver'], 'bo', alpha=0.2)
plt.show()


# In[ ]:


#lung vs brest
plt.figure(figsize=(20,8))
plt.xlabel("Femalebreast")
plt.ylabel("liver")
plt.suptitle("Joint distribution of lung vs Femalebreast", fontsize= 15)
plt.plot(df1['Lung'], df1['Femalebreast'], 'bo', alpha=0.2)
plt.show()


# In[ ]:


#lung vs leukemia
plt.figure(figsize=(20,8))
plt.xlabel("lung")
plt.ylabel("Leukemia")
plt.suptitle("Joint distribution of lung vs Leukemia", fontsize= 15)
plt.plot(df1['Lung'], df1['Leukemia'], 'bo', alpha=0.2)
plt.show()

#fig.set_size_inches(20,10)
#sns.scatterplot(x='Lung',y='Leukemia',data=df1, size = 20);


# # * Conclusion

# 1) THE DATA SET IS SLIGHTLY SKWED AND NOT A PERFECT DATASET. DUE TO LESS AMOUNT OF DATA I THINK SKEWNESS IS THERE.
# 
# 2) FROM OLS MODEL WE CAN INFER THAT THE DATA IS NOT HAVELIY PANALISED. WE CAN SEE THAT THERE IS GOOD MODEL IN BETWEEN LIVER, OVERY, PANCRES, PROSTATE CANCER.
# 
# 3) IN CANCER TYPE THE LUNG CANCER IS THE MOST COMMON FORM AMONG ALL THE CANCER, FOLLOWE BY PROSTATE, BREAST AND COLON CANCER.
# 
# 4) CALIFORNIA IS THE LEADING STATE IN ALL CANCER WHERE AS SOME TYPE OF CANCER CANT BE FOUND IN VERMONT. CALIFORNIA IS FOLLOWED BY FLORIDA, TEXAS AND NEW YORK.

# In[ ]:




