#!/usr/bin/env python
# coding: utf-8

# HAPPINES IN THE WORLD
# 
# ![](https://ichef.bbci.co.uk/news/660/cpsprodpb/1655A/production/_92928419_thinkstockphotos-508347326.jpg)
# 
# 
# 
# Hi there!
# Im gonna show you some vizualitations about the Happiness Report from 2015.
# In this Kernel i want to emphasize in the correlation of different variables with the Happiness Score and extract some insights.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../input/2015.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df['Country'].nunique()


# In[ ]:


df['Region'].nunique()


# In[ ]:


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

dfMap = dict(type = 'choropleth', locations = df['Country'],locationmode = 'country names',z = df['Happiness Score'], text = df['Country'],colorbar = {'title':'World Happiness Score'})
layout = dict(title = 'World Happiness Score', geo = dict(showframe = False))
WorldMap = go.Figure(data = [dfMap], layout=layout)
iplot(WorldMap)


# This plot has been adapted from the wonderful Kernel https://www.kaggle.com/sarahvch/investigating-happiness-with-python 

# In[ ]:


#Its very curious the STD
dfSTD = df.sort_values(by='Standard Error', axis=0, ascending=False)
dfSTD.head(10)


# Having the standard error of the happiness score we can see ordered wich countries present the biggest standard error. 
# If this score were greater, that could mean that the social differences in these countries are very large, so the overall Happiness Score wouldnt be reliable. But in our datafram, the standard error looks normal

# In[ ]:


dfCorr = df.drop(['Happiness Rank'],axis=1)


# In[ ]:


plt.subplots(figsize=(15,10))
sns.heatmap(dfCorr.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)
plt.title('Correlation of the dataset', size=25)
plt.show()


# In general, the highest correlation are between Happiness Score with Economy, Family and Health. 
# We can also find some correlation between Happiness Score with Freedom and Dystopia.
# I would like to recall that Trust and Generosity have the lowest correlation with the Happiness Score, this is something surprising for me.
# Its also interesting to see the correlation between Health and Economy but this is a different topic

# In[ ]:





# In[ ]:


sns.jointplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df, kind="reg",height=10, color="b")
sns.jointplot(x='Happiness Score',y='Family',data=df, kind="reg",height=10, color="b")
sns.jointplot(x='Happiness Score',y='Health (Life Expectancy)',data=df, kind="reg",height=10, color="b")


# I want to pack the data in continents.

# In[ ]:


df['Continent'] = df['Region']


# In[ ]:


df['Continent']=df['Continent'].apply(lambda x: x.replace('Western Europe', 'Europe'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Central and Eastern Europe', 'Europe'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Middle East and Northern Africa', 'Africa'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Sub-Saharan Africa', 'Africa'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Southeastern Asia', 'Asia'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Eastern Asia', 'Asia'))
df['Continent']=df['Continent'].apply(lambda x: x.replace('Southern Asia', 'Asia'))


# In[ ]:


df['Continent'].unique()


# In[ ]:


df.head()


# In[ ]:


dfCorrEurope = df.loc[df['Continent']=='Europe']
dfCorrEurope = dfCorrEurope.drop(['Happiness Rank'],axis=1)
plt.subplots(figsize=(15,10))
sns.heatmap(dfCorrEurope.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)
plt.title('Correlation of the dataset Europe', size=25)
plt.show()


# Here we can see something very interesting. In Europe there is a big correlation between Happiness and Freedom, and less correlation between Happiness and Health something that is not seen in the general correlation matrix

# In[ ]:


dfCorrAfrica = df.loc[df['Continent']=='Africa']
dfCorrAfrica = dfCorrAfrica.drop(['Happiness Rank'],axis=1)
plt.subplots(figsize=(15,10))
sns.heatmap(dfCorrAfrica.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)
plt.title('Correlation of the dataset Africa', size=25)
plt.show()


# In[ ]:


dfCorrSA = df.loc[df['Continent']=='Latin America and Caribbean']
dfCorrSA = dfCorrSA.drop(['Happiness Rank'],axis=1)
plt.subplots(figsize=(15,10))
sns.heatmap(dfCorrSA.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)
plt.title('Correlation of the dataset South America', size=25)
plt.show()


# In[ ]:


dfCorrAsia = df.loc[df['Continent']=='Asia']
dfCorrAsia = dfCorrAsia.drop(['Happiness Rank'],axis=1)
plt.subplots(figsize=(15,10))
sns.heatmap(dfCorrAsia.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)
plt.title('Correlation of the dataset Asia', size=25)
plt.show()


# We can see very interesting points in the specific correlation matrix per continents.
# The correlation with the Happiness is not always the same in differens continents

# In[ ]:


sns.set(rc={'figure.figsize':(18,9)})

sns.boxplot(x="Continent", y="Happiness Score", data=df)


# In[ ]:


sns.swarmplot(x="Continent", y="Happiness Score", data=df,palette="Set2", dodge=True)


# In[ ]:





# Here we can visualize that the data in Europe and Africa is very scattered so the average Happiness Score will be tricky.
# As we will see in some lines, in the top Happier Countries there is a lot of countries from Europe but it doesn't mean that is one of the happier countries. This is because the data is scattered and the biggest groups (countries) are between 5-6 points of the Score.

# In[ ]:


dfTop = df.head(12)
sns.barplot(x="Country", y="Happiness Score", data=dfTop, ci=68,  palette="Blues_d")


# In[ ]:


dfPerContinent = df.groupby(by='Continent')['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)'
                           ,'Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual'].mean()
dfPerContinent=dfPerContinent.reset_index()

dfPerContinentO = dfPerContinent.sort_values(by='Happiness Score',ascending=False)
sns.barplot(x="Continent", y="Happiness Score", data=dfPerContinentO, ci=68,  palette="Blues_d")


# In[ ]:


dfPerRegion = df.groupby(by='Region')['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)'
                           ,'Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual'].mean()

dfPerRegionO = dfPerRegion.sort_values(by='Happiness Score',ascending=False)
dfPerRegionO=dfPerRegionO.reset_index()
g = sns.barplot(x="Region", y="Happiness Score", data=dfPerRegionO, ci=68,  palette="Blues_d")
plt.xticks(rotation=90)


# As a citizen of Spain with roots of Poland, is very interesting for me to see how Western Europe occupies the third place while Central and Eastern Europe ocuppies the seventh place. In fact the difference is big, and that explains why Europe has the fourth place in the list of happiest continents in the world.

# In the next days im going to check this Kernel in order to improve it, i would be very happy if someone advises me.
# 
# Let's do some Linear Regression, we are gonna use all the features to predict the Happiness Score.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df1 = df.drop(columns=['Country','Happiness Rank','Region'])

X = df1.drop(columns=['Happiness Score'])
y = df1['Happiness Score']

X1 = np.array(X)
y1 = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state=42)

reg_all = LinearRegression()

reg_all.fit(X_train, y_train)

y_pred = reg_all.predict(X_test)

print("R^2: {}".format(reg_all.score(X_test, y_test)))


# With this R^2 the model explains all the variability of the response data around its mean so its a pretty good way to make our predictions .

# In[ ]:





# In[ ]:





# In[ ]:




