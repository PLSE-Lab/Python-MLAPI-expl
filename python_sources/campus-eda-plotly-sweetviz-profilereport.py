#!/usr/bin/env python
# coding: utf-8

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


from pandas_profiling import ProfileReport
get_ipython().system('pip install sweetviz')


# In[ ]:


pr = ProfileReport(df)


# In[ ]:


pr


# In[ ]:


import sweetviz
sweetviz_report = sweetviz.analyze([df,"data"],target_feat='salary')


# In[ ]:


sweetviz_report.show_html('viz.html')

# you can see the report in data section (output section) download it and view it


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df['gender'])


# In[ ]:


sns.countplot(df['specialisation'])


# In[ ]:


sns.countplot(df['hsc_s'])


# In[ ]:


sns.countplot(df['hsc_b'])


# In[ ]:


sns.countplot(df['degree_t'])


# In[ ]:


sns.countplot(df['workex'])


# In[ ]:


sns.countplot(df['status'])


# In[ ]:


sns.distplot(df['salary'])
plt.show()


# In[ ]:


salary_gen = df.groupby(["gender"])[["salary"]].mean().reset_index()
salary_gen


# In[ ]:





# In[ ]:


import plotly.express as px
fig = px.bar(salary_gen,x='gender',y='salary',color='gender',template='ggplot2')
fig.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu",linewidth=0.3)
plt.show()


# In[ ]:


grouped = df.groupby(["gender",'workex'])[["salary"]].mean().reset_index().sort_values(by='salary',ascending=True)
grouped


# In[ ]:


px.bar(grouped,x='gender',y='salary',color='workex')


# In[ ]:


grouped = df.groupby(["gender",'specialisation'])[["salary"]].mean().reset_index().sort_values(by='salary',ascending=True)
px.bar(grouped,x='gender',y='salary',color='specialisation',barmode='group')


# In[ ]:


px.histogram(df,x='degree_p',y='salary',color='workex')


# In[ ]:


px.density_heatmap(df,x='degree_p',y='salary',marginal_x='box',marginal_y='histogram')


# In[ ]:


px.density_heatmap(df,x='degree_p',y='salary',facet_row="ssc_b", facet_col="hsc_b")


# In[ ]:


df.head(2)


# In[ ]:


px.histogram(df,"ssc_p",'salary',color='gender',marginal='violin')


# In[ ]:



px.histogram(df,"ssc_p",'salary',color='gender',marginal='box')


# In[ ]:


sns.relplot(data=df,x='mba_p',y='salary',hue='gender',palette='gnuplot')


# In[ ]:


df.head(2)


# In[ ]:


group = df.groupby(['degree_t'])[['degree_p']].mean().reset_index()
fig = px.pie(group,values='degree_p',names='degree_t')
fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label")
fig.show()


# In[ ]:


df.head(2)


# In[ ]:


sns.catplot(x='gender',y='hsc_p',data=df)


# In[ ]:


sns.catplot(x='gender',y='ssc_p',data=df)


# In[ ]:


sns.violinplot(x='gender',y='hsc_p',data=df)


# In[ ]:


sns.violinplot(x='gender',y='ssc_p',data=df)


# In[ ]:


sns.violinplot(x='gender',y='salary',data=df)


# In[ ]:


sns.catplot(x='gender',y='salary',data=df)


# In[ ]:





# In[ ]:


sns.stripplot(x='gender',y='hsc_p',data=df)
sns.violinplot(x='gender',y='hsc_p',data=df)


# In[ ]:


sns.boxplot(x='gender',y='hsc_p',data=df)


# In[ ]:


sns.boxplot(x='gender',y='ssc_p',data=df)


# In[ ]:


sns.boxplot(x='gender',y='salary',data=df)


# In[ ]:


sns.stripplot(x='gender',y='hsc_p',data=df)
sns.boxplot(x='gender',y='hsc_p',data=df)


# In[ ]:


sns.swarmplot(x='hsc_s',y='hsc_p',data=df)


# In[ ]:


sns.clustermap(df.corr(),linewidth=0.3,cmap='summer',annot=True)


# In[ ]:


cor = df.loc[:,['hsc_p','ssc_p','salary']]
sns.clustermap(cor.corr(),annot=True,linewidth=0.6)


# In[ ]:


plt.figure(figsize=(10,6))
import plotly.express as px
ms = df.sort_values(by=['salary'],ascending=False)
px.funnel(ms,'salary','degree_t')


# In[ ]:


ms = df.sort_values(by=['salary'],ascending=False)
px.funnel(ms,'salary','specialisation')


# In[ ]:


df.head(2)


# In[ ]:


ms = df.sort_values(by=['salary'],ascending=False)
px.funnel(ms,'salary','hsc_s')


# In[ ]:


df.head()


# In[ ]:


# plt.figure(figsize=(10,6))
# sns.set_style("darkgrid")
# sns.jointplot(x=df["mba_p"], y=df["salary"], kind='scatter')

# plt.figure(figsize=(10,6))
# sns.set_style("darkgrid")
# sns.jointplot(x=df["mba_p"], y=df["salary"], kind='resid')

plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.jointplot(x=df["mba_p"], y=df["salary"], kind='hex',color='g')

# plt.figure(figsize=(10,6))
# sns.set_style("darkgrid")
# sns.jointplot(x=df["mba_p"], y=df["salary"], kind='kde')


# In[ ]:


plt.figure(figsize=(16,6))
sns.violinplot(x="degree_t", y="salary", hue="specialisation",data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.violinplot(x="degree_t", y="salary", hue="workex",data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.boxenplot(data=df,x='ssc_b',y='ssc_p',hue='gender')
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.boxenplot(data=df,x='hsc_b',y='hsc_p',hue='gender')
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.boxenplot(data=df,x='specialisation',y='mba_p',hue='gender')
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.boxenplot(data=df,x='ssc_b',y='ssc_p',hue='workex')
plt.show()


# In[ ]:


px.box(df,x="hsc_b", y="hsc_p",color = "workex")


# In[ ]:


px.box(df,x="ssc_b", y="ssc_p",color = "workex")


# In[ ]:


px.box(df,x="degree_t", y="salary",color = "workex",notched=True)


# In[ ]:


get_ipython().system('pip install bubbly')
get_ipython().system('pip install chart_studio')
from bubbly.bubbly import bubbleplot
from plotly.offline import iplot
import chart_studio.plotly as py


# In[ ]:


iplot(bubbleplot(df,x_column='hsc_p',y_column='salary',bubble_column='hsc_b'))


# In[ ]:


plt.figure(figsize=(20,7))
sns.lmplot(data=df,x='ssc_p',y='salary')
plt.show()


# In[ ]:


xyz = df
xyz.dropna(inplace=True)


# In[ ]:


fig = bubbleplot(xyz,x_column='hsc_p',y_column='salary',bubble_column='hsc_b',size_column='salary', color_column='workex',x_logscale=True, scale_bubble=2)
iplot(fig)


# In[ ]:


fig = bubbleplot(xyz,x_column='ssc_p',y_column='salary',bubble_column='ssc_b',size_column='salary', color_column='specialisation',x_logscale=True, scale_bubble=2)
iplot(fig)


# In[ ]:


fig = bubbleplot(xyz,x_column='ssc_p',y_column='salary',bubble_column='ssc_b',size_column='salary', color_column='degree_t',x_logscale=True, scale_bubble=2)
iplot(fig)


# In[ ]:


fig = px.scatter_ternary(df, a="ssc_p", b="hsc_p",c="degree_p",color = "degree_t")
fig.show()


# In[ ]:


fig = px.scatter_ternary(df, a="ssc_p", b="hsc_p",c="degree_p",color = "workex")
fig.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.residplot(data=df,x='ssc_p',y='salary')
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.residplot(data=df,x='hsc_p',y='salary')
plt.show()


# In[ ]:


df.plot.area(y=['ssc_p','hsc_p','degree_p'],alpha=0.6,figsize=(20, 6))


# In[ ]:


df.plot.area(y=['mba_p','hsc_p','degree_p'],alpha=0.6,figsize=(20, 6))


# In[ ]:


wind = px.data.wind()
wind.head()


# In[ ]:


fig = px.bar_polar(wind, r="frequency", theta="direction",color="strength")
fig.show()


# In[ ]:


fig = px.line_polar(wind, r="frequency", theta="direction",color="strength")
fig.show()


# In[ ]:


fig = px.scatter_polar(wind, r="frequency", theta="direction",color="strength")
fig.show()


# In[ ]:


px.scatter(df, x="hsc_s", y="hsc_p", color="specialisation")


# In[ ]:


px.scatter(df, x="ssc_b", y="ssc_p", color="specialisation")


# # Model

# In[ ]:


df.head(2)


# In[ ]:


df.drop(columns=['sl_no'],inplace=True)
df.head(2)


# In[ ]:


categorical = df.columns[df.dtypes=='object']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[ ]:


categorical


# In[ ]:


df[categorical]=df[categorical].apply(encoder.fit_transform)


# In[ ]:


df.head()


# In[ ]:


df['salary'] = df['salary'].fillna(0)


# In[ ]:


df.isnull().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


X = df.drop(columns=['salary'])
y = df['salary']


# In[ ]:


X_scl = scaler.fit_transform(X)
X_scl


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scl,y,test_size=0.2)


# In[ ]:


X_train.shape, y_train.shape,X_test.shape,y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
mean_squared_error(y_pred,y_test)


# In[ ]:




