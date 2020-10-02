#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Disabling warnings
import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv('../input/master.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample(10)


# In[ ]:


df.info()


# In[ ]:


df.rename(columns={'HDI for year': 'HDI_for_year', 'country-year':'country_year', 'suicides/100k pop': 'suicides/100k_pop', ' gdp_for_year ($) ':'gdp_for_year', 'gdp_per_capita ($)':'gdp_per_capita'}, inplace=True);


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(1, inplace=True);


# In[ ]:


df.isnull().any()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.index = df.year


# In[ ]:


df.head(5)


# In[ ]:


data_per_year = df.groupby('year').sum()
data_per_year


# In[ ]:


# Plot Total No. Of Suicides Per Year From 1985 To 2016.
data_per_year['suicides_no'].plot()
plt.title('Total No. Of Suicides Per Year From 1985 To 2016')
plt.ylabel('No. Suicides')
plt.xlabel('Year')
plt.xlim((df.year.min() - 1), (df.year.max() + 1))
plt.show()


# In[ ]:


df.groupby(['year', 'sex']).suicides_no.sum();


# In[ ]:


# Polt Total No. Of Suicides Per Year From 1985 To 2016 Hue To Gendar.
df.pivot_table('suicides_no', index='year', columns='sex', aggfunc='sum').plot()
plt.title('Total No. Of Suicides Per Year From 1985 To 2016 Hue To Gendar')
plt.ylabel('No. Suicides')
plt.xlabel('Year')
plt.xlim((df.year.min() - 1), (df.year.max() + 1))
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='year', y='suicides_no', data=df, hue='sex') 
plt.title('Total No. Of Suicides Per Year From 1985 To 2016 Hue To Gendar')
plt.ylabel('No. Suicides')
plt.xlabel('Year')
plt.xlim((df.year.min() - 1), (df.year.max() + 1))
plt.show()


# In[ ]:


df.columns


# In[ ]:


# Bar Plot No. of Suicides per country last 30 years.

# Main Var. for Ploting
sui_no = df.groupby(['country']).suicides_no.sum()
countries = []
for (i, m) in df.groupby('country'):
    countries.append(i)
countries = np.array(countries);

# ploting
plt.figure(figsize=(10,20))
sns.barplot(y=countries, x=sui_no)
plt.xlabel('No. of suicides')
plt.ylabel('Countries')
plt.title('Total No. of Suicides per country from 1987 to 2016')
plt.xlim(0, 1e6)
plt.show()


# In[ ]:


# Bar Plot No. of Suicides per Age last 30 years.

# Set Variables.
age_sui = df.pivot_table('suicides_no', index='age', aggfunc='sum')
x = age_sui.index.values
y = age_sui.values
y = y.reshape(6,)

# Ploting
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=x, y=y)
ax.set(title='No. Of Suicides Per Age', ylabel='No. of suicides', xlabel='Age');
plt.xticks(rotation=45);
plt.show()


# In[ ]:


gen_sui = df.pivot_table('suicides_no', index='generation', aggfunc='sum')
x = gen_sui.index.values
y = gen_sui.values
y = y.reshape(6,)

fig, ax = plt.subplots(figsize=(10, 6))
explode = (0.1,0.1,0.1,0.5,0.1,0.1)
ax.pie(y, explode=explode, labels=x, autopct='%1.1f%%', shadow=True, startangle=0)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.stripplot(x="year", y='suicides/100k_pop', data=df)
plt.title('No. Of Suicides/100k Population')
plt.xlabel('Year')
plt.ylabel('Suicides/100k Population')
plt.xticks(rotation=60)
plt.show()


# In[ ]:


sns.distplot(df['suicides/100k_pop'])
plt.show()


# In[ ]:


sns.set_color_codes()
sns.distplot(df['country'].value_counts().values)
plt.show()


# In[ ]:


sns.pairplot(df, hue="sex")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split


# In[ ]:


data = df.copy()
age_sui


# In[ ]:


data.generation.replace(['Boomers', 'Generation X', 'Generation Z', 'G.I. Generation', 'Millenials', 'Silent'], 
                        ['0', '1', '2', '3', '4', '5'], inplace=True)

data.sex.replace(['male', 'female'], ['0', '1'], inplace=True)

data['gdp_for_year'] = data['gdp_for_year'].str.replace(',','')

def means(arr):
    return str(np.array(arr).mean())
data.age.replace(['15-24 years', '25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'], 
                 [means([15, 24]), means([25, 34]), means([35, 54]), 
                  means([5, 14]), means([55, 74]), means([75])], inplace=True)


# In[ ]:


data.drop(['country', 'year', 'country_year'], 1, inplace=True)


# In[ ]:


pd.to_numeric(data['generation']);
pd.to_numeric(data['sex']);
pd.to_numeric(data['gdp_for_year']);


# In[ ]:


#Correlation matrix & Heatmap
plt.figure(figsize =(10,8))
corrmat = data.corr()
sns.heatmap(corrmat, square=True, annot=True, cbar=True);


# In[ ]:


data['fatality_rate'] = np.where(data['suicides/100k_pop']>data['suicides/100k_pop'].mean(), 
                                 1, 0)


# In[ ]:


data.head()


# In[ ]:


X = np.array(data.drop(['fatality_rate', 'suicides/100k_pop'], 1))
y = np.array(data.fatality_rate)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


print("Model accuracy is: {0:.2f}".format(accuracy_score(y_test, y_pred) * 100))


# In[ ]:


mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat, square=True, annot=True, cbar=True)

plt.xlabel('predicted value')
plt.ylabel('true value');


# In[ ]:


print(classification_report(y_pred, y_test))


# In[ ]:




