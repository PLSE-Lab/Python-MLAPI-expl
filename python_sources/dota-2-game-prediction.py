#!/usr/bin/env python
# coding: utf-8

# Let's have some fun predicting some DOTA 2 games based on picks.

# <a id='top'></a>
# # Contents:
# **1. [Import](#import)** <br>
#     - Import Libraries
#     - Import Dataset
# **2. [Meet & Greet](#meeting)** <br>
#     - Knowing the Type of Data
#     - Missing Values
#     - Unique Values
#     - First Rows of the Dataset
# **3. [Explanatory Data Analysis (EDA)](#explanatory)** <br>
#     - Distribuition of Target Variable
#     - Identifying Bias

# <a id='import'></a> <br>
# # **1. Import**
# - Import Libraries
# - Import Dataset

# In[ ]:


#Loading the libraries
import numpy as np #Math library
import pandas as pd #Dataset library
import seaborn as sns #Graph library
import matplotlib.pyplot as plt #Help seaborn

#Importing data and renaming columns for consistency
df = pd.read_csv('../input/dota_games.txt', header=None)
df = df.rename(columns={0: 'ancient_1', 1: 'ancient_2', 2: 'ancient_3', 3: 'ancient_4', 4: 'ancient_5',
                        5: 'dire_1', 6: 'dire_2', 7: 'dire_3', 8: 'dire_4', 9: 'dire_5', 
                    10: 'team_win'})


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# <a href='#top'>back to top</a>

# <a id='meeting'></a> <br>
# # **2. Meet & Greet**
# - Knowing the Type of Data
# - Missing Values
# - Unique Values
# - First Rows of the Dataset

# Test if there any missing values in DataFrame `df`. It turns out there are no aparent missing values.

# In[12]:


print(df.info())


# In[13]:


#Looking unique values
print(df.nunique())


# In[14]:


#Knowing the data
print(df.head())


# <a href='#top'>back to top</a>

# <a id='explanatory'></a> <br>
# # **3. Explanatory Data Analysis (EDA)**
# - Distribuition of Target Variable
# - Identifying Bias

# ## Let's start looking through target variable and their distribuition

# In[17]:


import plotly.offline as py #library that implement interactive visuals
py.init_notebook_mode(connected=True) #allow us to work with offline plotly
import plotly.graph_objs as go #like "plt" of matplotlib
import plotly.tools as tls #it will be useful soon

trace0 = go.Bar(
    x = df[df['team_win'] == 1]['team_win'].value_counts().index.values,
    y = df[df['team_win'] == 1]['team_win'].value_counts().values,
    name = 'Ancient team'
)

trace1 = go.Bar(
    x = df[df['team_win'] == 2]['team_win'].value_counts().index.values,
    y = df[df['team_win'] == 2]['team_win'].value_counts().values,
    name = 'Dire team'
)

data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(title='Wins'),
    xaxis=dict(title='Team'),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# <h1>I'M REWRITING THIS KERNEL. AFTER THIS LINE IT'S JUST PAST WORK</h1>

# Split the data into features `X` and target `y`.

# In[ ]:


y = df['team_win']
X = df.drop(['team_win'], axis=1)


# <a href='#top'>back to top</a>

# #### Explanatory Data Analysis
# 
# Get to know our dataset is a succint way to gain insights.

# ##### 21. There is a team side advantage? The map influences the victory?
# 

# In[ ]:


print('The number of wins are equal to each team? {}'.format(
    round(len(df.loc[df.team_win == 1])/len(df.loc[df.team_win == 2]), 1) == 1
))
print('How much is this advantage ratio? {}%'.format(
    round(len(df.loc[df.team_win == 1])/len(df.loc[df.team_win == 2]) - 1, 3) * 100
))


# As we saw, there is a little advantage, but how we gonna balance it? Let's take a look of hero biggests win ratio.

# ##### 22. Which are the heroes with the biggest win ratio?

# In[ ]:


winners_team_1 = df.loc[df.team_win == 1][['ancient_1', 'ancient_2', 'ancient_3', 'ancient_4', 'ancient_5']]
winners_team_1.rename(index=str, inplace=True, columns={'ancient_1': 'player_1', 
                                                        'ancient_2': 'player_2', 
                                                        'ancient_3': 'player_3', 
                                                        'ancient_4': 'player_4', 
                                                        'ancient_5': 'player_5'})

winners_team_2 = df.loc[df.team_win == 2][['dire_1', 'dire_2', 'dire_3', 'dire_4', 'dire_5']]
winners_team_2.rename(index=str, inplace=True, columns={'dire_1': 'player_1',
                                                        'dire_2': 'player_2',
                                                        'dire_3': 'player_3',
                                                        'dire_4': 'player_4',
                                                        'dire_5': 'player_5'})

winners = winners_team_1.append(winners_team_2)

hero_wins = winners.player_1.value_counts() +             winners.player_2.value_counts() +             winners.player_3.value_counts() +             winners.player_4.value_counts() +             winners.player_5.value_counts()

hero_wins = hero_wins.sort_values(ascending=False)

# TODO: get the wins for each hero


# <a id='encoding'></a>
# #### 2. Encoding labels
# 
# Since the hero names are present in the feature data, we need to encode these names into values so the `sklearn` could work properly. Lets initialize the encoder with `LabelEncoder` that already been imported in previous section.

# In[ ]:


le = LabelEncoder()

for col in X.columns.values:
    le.fit(X[col].values)
    X[col] = le.transform(X[col])
    
print(X.info())


# <a href='#top'>back to top</a>

# <a id='pipeline'></a>
# #### 3. Putting into pipeline
# 
# I'll use pipeline because it's easiler to piece everything together (I could have put section 2 here, but... well, I had already written that section). Here, the pipeline includes scaling and hyperparameter tuning to classify victory. If you aren't familiar with these concepts, check this [course](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn) of Datacamp (sign in with a Microsoft account and have free trial for 2 months).
# 
# 

# In[ ]:


steps = [('scaler', StandardScaler()), ('logistic', SGDClassifier())]

pipeline = Pipeline(steps)


# We initialized the `Pipeline` class with the following steps: 
# * `StandardScaler` to normalize the data
# * `SGDClassifier` is our classifier model

# <a href='#top'>back to top</a>

# <a id='tuning'></a>
# ##### 31. Hyperparameter tuning
# 
# The hyperparameter we will tune is `alpha`. `alpha` controls the regularization strength. Check [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) to learn more about it. 
# 
# To tune, we need to specify a dictionary as the following pattern: `model__param`. Check out below.

# In[ ]:


alpha_space = np.logspace(-5, 8, 11)
param_grid = {'logistic__alpha': alpha_space}

cv = GridSearchCV(pipeline, param_grid, cv=5)


# <a href='#top'>back to top</a>

# <a id='training'></a>
# ##### 32. Training and predicting
# 
# Finally, train our model with the best parameters, but first, split our dataset with `train_test_split`. Remember to train with the training dataset.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv.fit(X_train, y_train)


# Now, predict the labels of the test set.

# In[ ]:


y_pred = cv.predict(X_test)


# <a href='#top'>back to top</a>

# <a id='evaluating'></a>
# #### 4. Evaluating model performance
# 
# Let's see the scores of the current model.

# In[ ]:


print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Tuned Model Score: {}".format(cv.best_score_))

