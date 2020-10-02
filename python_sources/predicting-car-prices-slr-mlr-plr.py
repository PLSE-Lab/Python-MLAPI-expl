#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[11]:


import pandas as pd
df = pd.read_csv("../input/Document.csv", header = None)


# In[12]:


df.head(10)


# In[13]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
headers


# In[15]:


df.columns = headers
df


# In[16]:


df.dropna(subset=["price"], axis = 0)
df


# In[17]:


df.dtypes


# In[18]:


df.describe()


# In[19]:


df.describe(include = "all")


# In[20]:


df.info


# In[21]:


import numpy as np
df.replace("?", np.NaN, inplace = True)
df.head(5)


# In[ ]:


missing_data = df.isnull()
missing_data.head(3)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    
    print(missing_data[column].value_counts())
    print(" ")
    


# In[ ]:


avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)
df["normalized-losses"].replace(np.NaN, avg_1, inplace = True)


# In[ ]:


avg_2 = df["bore"].astype("float").mean(axis = 0)
df["bore"].replace(np.NaN, avg_2, inplace = True)


# In[ ]:


avg_3 = df["stroke"].astype("float").mean(axis = 0)
df["stroke"].replace(np.NaN, avg_3, inplace = True)


# In[ ]:


avg_4 = df["horsepower"].astype("float").mean(axis = 0)
df["horsepower"].replace(np.NaN, avg_4, inplace = True)


# In[ ]:


avg_5 = df["peak-rpm"].astype("float").mean(axis = 0)
df["peak-rpm"].replace(np.NaN, avg_5, inplace = True)


# In[ ]:


df["num-of-doors"].value_counts()


# In[ ]:


df["num-of-doors"].value_counts().idxmax()


# In[ ]:


df["num-of-doors"].replace(np.NaN, "four", inplace = True)


# In[ ]:


df.dropna(subset = ["price"], axis = 0, inplace = True)
df.reset_index(drop = True,inplace = True)


# In[ ]:


df


# In[ ]:


df.dtypes


# In[ ]:


df[["bore", "stroke", "peak-rpm"]] = df[["bore", "stroke", "peak-rpm"]].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype(float)


# In[ ]:


df.dtypes


# In[ ]:


df["city L/100km"] = 235/df["city-mpg"]
df["highway L/100km"] = 235/df["highway-mpg"]


# In[ ]:


df.head(5)


# In[ ]:


df["horsepower"] = df["horsepower"].astype(float)


# In[ ]:


binwidth = (max(df["horsepower"]) - min(df["horsepower"]))/4


# In[ ]:


bins  = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
bins


# In[ ]:


group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels = group_names,include_lowest = True)
df['horsepower-binned'].head(5)


# In[ ]:


df.head(10)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


plt.pyplot.hist(df["horsepower"], bins = 3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("hp bins")


# In[ ]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[ ]:


dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()


# In[ ]:


df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis =1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


dummy_variable2 = pd.get_dummies(df['aspiration'])
dummy_variable2.rename(columns = {'std': 'aspiration-std' , 'turbo' : 'aspiration-turbo'}, inplace = True)
df = pd.concat([df,dummy_variable2], axis = 1)
df.drop("aspiration", axis =1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.to_csv('clean_df.csv')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.dtypes


# In[ ]:


df.corr()


# In[ ]:


df[['bore','stroke','compression-ratio','horsepower']].corr()


# In[ ]:


sns.regplot(x = "engine-size", y = "price", data = df)
plt.ylim(0,)


# In[ ]:


df[["engine-size","price"]].corr()


# In[ ]:


sns.regplot(x = "highway-mpg", y ="price", data = df)
plt.ylim(0,)
df[["highway-mpg", "price"]].corr()


# In[ ]:


sns.regplot(x = "peak-rpm", y = "price", data = df)
plt.ylim(0,)
df[["peak-rpm", "price"]].corr()


# In[ ]:



sns.regplot(x="stroke",y = "price", data = df)
plt.ylim(0,)
df[["stroke","price"]].corr()


# In[ ]:


sns.boxplot(x= "body-style", y = "price", data = df)
plt.ylim(0,)


# In[ ]:


sns.boxplot(x = "engine-location", y = "price", data = df)


plt.ylim(0,)


# In[ ]:


sns.boxplot(x = "drive-wheels", y = "price", data = df)
plt.ylim(0,)


# In[ ]:


df.describe(include=['object'])


# In[ ]:


df['drive-wheels'].value_counts()


# In[ ]:


dw_count = df['drive-wheels'].value_counts().to_frame()
dw_count.rename(columns={'drive-wheels': 'count'}, inplace = True)
dw_count.index.name = 'drive-wheels'
dw_count


# In[ ]:


el_count = df['engine-location'].value_counts().to_frame()
el_count.rename(columns={'engine-location':'count'}, inplace = True)
el_count.index.name = 'engine-location'
el_count


# In[ ]:


df['drive-wheels'].unique()


# In[ ]:


df_group_one = df[['drive-wheels','body-style', 'price']]
df_group_one = df_group_one.groupby(['drive-wheels','body-style'], as_index = False).mean()
df_group_one


# In[ ]:


pivot = df_group_one.pivot(index='drive-wheels', columns ='body-style')
pivot


# In[ ]:


pivot = pivot.fillna(0)
pivot


# In[ ]:


df['body-style'].unique()


# In[ ]:


df_group_two = df[['body-style', 'price']]
df_group_two = df_group_two.groupby(['body-style'], as_index = False).mean()
df_group_two


# In[ ]:


pivot2 = df_group_two.pivot(columns = 'body-style')
pivot2 = pivot2.fillna(0)
pivot2


# In[ ]:


plt.pcolor(pivot,cmap= 'RdBu')
plt.colorbar()
plt.show()


# In[ ]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
pearson_coef, p_value


# In[ ]:


df_anova = df[["drive-wheels","price"]]
grouped_anova = df_anova.groupby(['drive-wheels'])
grouped_anova.head(1)


# In[ ]:


grouped_anova.get_group('4wd')['price']


# In[ ]:


f_val, p_val = stats.f_oneway(grouped_anova.get_group('4wd')['price'], grouped_anova.get_group('rwd')['price'])
f_val, p_val


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat = lm.predict(X)
Yhat[0:5]


# In[ ]:


print(lm.intercept_)
print(lm.coef_)


# In[ ]:


lm1 = LinearRegression()
lm1.fit(df[['engine-size']], df['price'])
print(lm1.intercept_)
print(lm1.coef_)


# In[ ]:


Z = df[['horsepower', 'curb-weight', 'engine-size','highway-mpg']]
lm.fit(Z, df['price'])
Y_hat = lm.predict(Z)


# In[ ]:


lm.fit(df[['normalized-losses', 'highway-mpg']], df['price'])
print(lm.intercept_)
print(lm.coef_)


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


width = 12
height  = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y ="price", data = df)
plt.ylim(0,)


# In[ ]:


plt.figure(figsize=(12,10))
sns.regplot(x='peak-rpm', y ='price', data = df)
plt.ylim(0,)


# In[ ]:


df[["peak-rpm", "highway-mpg", "price"]].corr()


# In[ ]:



plt.figure(figsize=(12,10))
sns.residplot(df['highway-mpg'], df['price'])
plt.ylim(0,)


# In[ ]:


sns.residplot(df['peak-rpm'],df['price'])
plt.ylim(0,)


# In[ ]:



Y_hat = lm.predict(Z)


# In[ ]:



plt.figure(figsize=(12,10))
ax1 = sns.distplot(df[['price']], hist=False, color ='r', label = 'Actual Values')
sns.distplot(Y_hat, hist=False, color = 'b', label = 'fitted values',ax= ax1)
plt.title('Actual vs fitted')
plt.xlabel('Price')
plt.ylabel('Proportion in cars')


# In[ ]:


def PlotPolly(model, independent_variable, dependent_variable, name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable,'.', x_new, y_new, '-')
    plt.title('Polynomial fit')
    ax = plt.gca()
    ax.set_facecolor((0.88, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(name)
    plt.ylabel('price')
    plt.show()


# In[ ]:


x = df['highway-mpg']
y = df['price']


# In[ ]:


f = np.polyfit(x,y,3)
p = np.poly1d(f)


# In[ ]:


PlotPolly(p,x,y,'highway-mpg')


# In[ ]:


f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'highway-mpg')


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


pr = PolynomialFeatures(degree = 2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
Z_pr.shape


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]


# In[ ]:


pipe = Pipeline(Input)
pipe


# In[ ]:


pipe.fit(Z,y)


# In[ ]:


ypipe = pipe.predict(Z)


# In[ ]:


ypipe[0:5]


# In[ ]:


lm.fit(X,Y)
lm.score(X,Y)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


Yhat = lm.predict(X)
mean_squared_error(df['price'], Yhat)


# In[ ]:


lm.fit(Z,Y)
lm.score(Z,Y)


# In[ ]:


Y_hat = lm.predict(Z)

mean_squared_error(df['price'], Y_hat)


# In[ ]:


from sklearn.metrics import r2_score
r_squared = r2_score(y,p(x))
print(r_squared)
mean_squared_error(y,p(x))


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

new_input = np.arange(1,100,1).reshape(-1,1)
lm.fit(X,Y)
yhat = lm.predict(new_input)
print(yhat[0:5])


plt.plot(new_input,yhat)


# In[ ]:




