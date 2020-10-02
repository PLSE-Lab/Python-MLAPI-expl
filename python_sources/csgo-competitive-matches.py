#!/usr/bin/env python
# coding: utf-8

# # Counter Strike Global Offensive Linear Regression Model
# 
# Counter Strike: Global Offensive is a first-person shooter video game in which two teams play as terrorists and counter-terrorists, respectively, and compete to blow up or defend an objective. Teams are five people per side, with players acquiring new weapons and items using points earned during the rounds in between rounds of play. Whichever team wins enough rounds, either by killing the entire enemy team or by successfully planting or defusing the bomb, wins the match.
# 
# The data used has been scraped by me and consists of all the competitive matches I have played. If you want to use your own data, use an extension called Ban Checker For Steam and then go to your competitive matches history, after that load all your data and save the webpage somewhere. Use the [Script](https://github.com/reedkihaddi/MachineLearning/blob/master/LinearRegression/CSGO/get_Data.py) to scrape the data and save it to a CSV file.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/mycsgo-data/model_data.csv')
df


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


from scipy import stats
z = np.abs(stats.zscore(df.iloc[:,1:]))
df = df[(z < 3).all(axis=1)]
df


# In[ ]:


plt.style.use('fivethirtyeight')
sns.kdeplot(df['Kills'])
plt.title("Kills")
plt.show()


# In[ ]:


sns.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 8})


# In[ ]:


sns.pairplot(df)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
y = df.iloc[:, 2]
x = df.iloc[:, [1, 3, 4, 5, 6]]
scaler = StandardScaler()
scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# # Regression

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
model = LinearRegression(normalize = True)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(model.score(x_test, y_test))
print(mean_squared_error(y_pred, y_test))


# In[ ]:


sns.residplot(y_test, y_pred, color="orange", scatter_kws={"s": 2})
plt.show()

