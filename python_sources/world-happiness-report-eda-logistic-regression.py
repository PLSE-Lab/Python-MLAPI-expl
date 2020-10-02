#!/usr/bin/env python
# coding: utf-8

# # 1.Data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv("../input/world-happiness/2017.csv", delimiter=',')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


print(df.head())


# # 2. EDA

# In[ ]:


df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',
      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(18,12), bins=50, grid=False);


# In[ ]:


sns.jointplot(x='Happiness.Score',y='Freedom',data=df,kind='scatter');


# In[ ]:


sns.pairplot(df);


# In[ ]:


df.corr()


# In[ ]:


plt.subplots(figsize=(10,8))
sns.heatmap(df.corr());


# In[ ]:


plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True);


# In[ ]:


plt.style.use('dark_background')
df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',
      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(20, 15), bins=50, grid=False);


# In[ ]:


plt.style.use('ggplot')
df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',
      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(20, 15), bins=50, grid=False);


# In[ ]:


plt.style.use('fivethirtyeight')
df.plot.area(alpha=0.4);


# # 3.MODEL

# In[ ]:


cat_feats = ['Country']

final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.info


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Country',axis=1), 
                                                    df['Country'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
             
             


# # 4. Predictions

# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:




