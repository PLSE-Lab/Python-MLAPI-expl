#!/usr/bin/env python
# coding: utf-8

# Exploration of How Social Media Can Predict Winning Metrics Better Than Salary

# In[ ]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


nba_salary = pd.read_csv("../input/nba_2017_nba_players_with_salary.csv")
nba_salary.head()

table = nba_salary[["PLAYER","POINTS","SALARY_MILLIONS"]]
table.plot.scatter(x='POINTS', y='SALARY_MILLIONS')

#Seems like there is a positive relationship between Points VS. Salary


# In[ ]:


nba_salary.describe()


# In[ ]:


nba_salary.head()

# Position, 
# age,
# MPG
# TEAM
# MPG
# W
# Salary

table2 = nba_salary[["PLAYER","POSITION","AGE","MPG","TEAM","W","POINTS","SALARY_MILLIONS"]]

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season")
corr = table2.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#as we can see: Points VS Salary; and MPG VS Salary have positive corelation 
#while the number of wins and age have no correlation with individual play's salary


# In[ ]:


sns.lmplot(x="SALARY_MILLIONS", y="POINTS", data=table2)


# In[ ]:


sns.lmplot(x="SALARY_MILLIONS", y="MPG", data=table2)


# In[ ]:


results = smf.ols('SALARY_MILLIONS ~POINTS', data=table2).fit()
print(results.summary())


# In[ ]:


table2.head()


# In[ ]:


table3 = table2[["PLAYER","W","POINTS","SALARY_MILLIONS","MPG"]]
median_point_player = table3.groupby("PLAYER").median()
median_point_player.head()


# In[ ]:


table4 = table2[["POSITION","W","POINTS","SALARY_MILLIONS","MPG","AGE"]]
median_point_position = table4.groupby("POSITION").median()
median_point_position.head()


# In[ ]:


#median_point['SALARY_MILLIONS'].value_counts().plot.bar()
#sns.distplot(median_point['SALARY_MILLIONS'], bins=10, kde=False)
#?sns.distplot()

import matplotlib.pyplot as plt; 
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('C', 'PF', 'PF-C', 'PG', 'SF')
y_pos = np.arange(len(objects))
performance = [5,5.56,1.61,4.77,4.74]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Salary_In_Millions')
plt.title('Position')
 
plt.show()


# In[ ]:




