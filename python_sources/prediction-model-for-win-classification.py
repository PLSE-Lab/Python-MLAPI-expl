#!/usr/bin/env python
# coding: utf-8

# <h1>Prediction Model for Win Classification in League of Legends Matches</h1>

# In[ ]:


import numpy as np
import pandas as pd

dataset = pd.read_csv("/kaggle/input/league-of-legends-ranked-games/challenger.csv")
dataset.drop(['vilemaw_kills_team_1', 'vilemaw_kills_team_2'], axis=1, inplace=True)
dataset.head(10)


# <h2>1. Data analysis(Challenger's Games)</h2>

# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# <h3>1.1. Win Distribution</h3>

# In[ ]:


g = sns.distplot(dataset["win"])


# <h3>1.2. Objectives Achievement</h3>

# In[ ]:


g = sns.lmplot(x="tower_kills_team_2", y="tower_kills_team_1", col="first_tower", height=4, data=dataset, hue="win")


# In[ ]:


g = sns.lmplot(x="inhibitor_kills_team_2", y="inhibitor_kills_team_1", col="first_inhibitor",data=dataset, height=4, hue="win")


# In[ ]:


g = sns.lmplot(x="dragon_kills_team_2", y="dragon_kills_team_1", col="first_dragon", data=dataset, height=4, hue="win")


# In[ ]:


g = sns.lmplot(x="baron_kills_team_1", y="baron_kills_team_2", col="first_baron", height=4, data=dataset, hue="win",  markers=["o", "x"])


# In[ ]:


g = sns.lmplot(x="rift_herald_kills_team_2", y="rift_herald_kills_team_1", col="first_rift_herald", data=dataset, height=4, hue="win", markers=["o", "x"])


# In[ ]:


plt.figure(figsize=(15,8))
plt.title("Relationship between objectives achievement factors and the team that won")
sns.heatmap(
    dataset.iloc[:,0:17].corr(),
    annot=False,
    linewidths=.5,
)


# <h3>1.3. Player's Performance</h3>

# In[ ]:


playersP = dataset.iloc[:,17:52]
playersP['win'] = dataset['win']

plt.figure(figsize=(15,8))
plt.title("Correlation between the performance of team 1 players")
sns.heatmap(
    playersP.corr(),
    annot=False,
    linewidths=.5,
)


# In[ ]:


playersP = dataset.iloc[:,52:88]
playersP['win'] = dataset['win']

plt.figure(figsize=(15,8))
plt.title("Correlation between the performance of team 2 players")
sns.heatmap(
    playersP.corr(),
    annot=False,
    linewidths=.5,
)


# In[ ]:


top=["gold_earned_20m_top_team_1", "cs_20m_top_team_1", "xp_20m_top_team_1", "damege_taken_20m_top_team_1"]
top2=["gold_earned_20m_top_team_2", "cs_20m_top_team_2", "xp_20m_top_team_2", "damege_taken_20m_top_team_2"]
middle=["gold_earned_20m_middle_team_1", "cs_20m_middle_team_1", "xp_20m_middle_team_1", "damege_taken_20m_middle_team_1"]
jungle=["gold_earned_20m_jungle_team_1", "cs_20m_jungle_team_1", "xp_20m_jungle_team_1", "damege_taken_20m_jungle_team_1"]
bottom_duo_carry=["gold_earned_20m_bottom_duo_carry_team_1", "cs_20m_bottom_duo_carry_team_1", "xp_20m_bottom_duo_carry_team_1", "damege_taken_20m_bottom_duo_carry_team_1"]
bottom_duo_support=["gold_earned_20m_bottom_duo_support_team_1", "cs_20m_bottom_duo_support_team_1", "xp_20m_bottom_duo_support_team_1", "damege_taken_20m_bottom_duo_support_team_1"]


# In[ ]:


g = sns.pairplot(dataset, vars=top, hue="win", height=3)


# In[ ]:


g = sns.pairplot(dataset, vars=middle, hue="win", height=3)


# In[ ]:


g = sns.pairplot(dataset, vars=jungle, hue="win", height=3)


# In[ ]:


g = sns.pairplot(dataset, vars=bottom_duo_carry, hue="win", height=4)


# In[ ]:


g = sns.pairplot(dataset, vars=bottom_duo_support, hue="win", height=4)


# <h2>2. Creating the model</h2>

# In[ ]:


x = dataset.drop('win', axis=1)
y = dataset['win']


# In[ ]:


print(x.shape)
print(y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

#divides the training dataset into training and testing, separating 25% in testing
#x = attributes e y = classes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[ ]:


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#create the tree
tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=5, random_state=0)
#create the model
model = tree.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
# prediction of test data 
predict = model.predict(x_test)


# In[ ]:


acc = accuracy_score(y_test, predict)
print("Accuracy: ", format(acc))


# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus as pydot

dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# <h2>3. Comparing results with the random game dataset</h2>

# In[ ]:


test = pd.read_csv("/kaggle/input/league-of-legends-ranked-games/others_tiers.csv")
test.drop(['vilemaw_kills_team_1', 'vilemaw_kills_team_2'], axis=1, inplace=True)
test.head(5)


# In[ ]:


g = sns.distplot(test["win"], kde=False)


# In[ ]:


x_test_random = test.drop('win', axis=1)
y_test_random = test['win']
print(x_test_random.shape)
print(y_test_random.shape)


# In[ ]:


predict_test = model.predict(x_test_random)
acc_test = accuracy_score(y_test_random, predict_test)
print("Accuracy: ", format(acc_test))

