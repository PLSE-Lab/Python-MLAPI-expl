#!/usr/bin/env python
# coding: utf-8

# # Pokemon data exploration

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
plt.style.use('ggplot')


# In[ ]:


df = pd.read_csv(r"../input/Pokemon.csv")


# --- 
# ## Visualization

# Let's visualize first the different types.

# In[ ]:


count = df.groupby('Type 1').size()
count.plot.bar(color='black', figsize=(15, 7))
plt.title('Occurences of Type 1')
plt.show()


# When it comes to single typed pokemon, the trend seems to stay the same.

# In[ ]:


singles = df[['Type 1', 'Type 2']].fillna('None').groupby(['Type 2', 'Type 1']).size()['None']
singles.plot.bar(color='black', figsize=(15, 7))
plt.title('Single typed pokemon Type')
plt.show()


# As for double typed pokemon, some of the most represented combos are: Normal-Flight, Bug-Flying, Grass-Poison & Bug-Poison 

# In[ ]:


count = df.groupby(['Type 1', 'Type 2']).size().unstack()
fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(count, xticklabels=count.columns, 
            yticklabels=count.columns,
           cmap='plasma')
plt.title('Double type combinations')
plt.show()


# Let's visualize the relationship between Attack & Defense, and Special Attack & Special Defense. 
# 
# It seems that there is some correlation between these two couples of variable: A high attack may imply a high defense. 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
df.plot.hexbin(x='Attack', y='Defense', gridsize=50, ax=ax1)
ax1.set_title('Attack - Defense')
df.plot.hexbin(x='Sp. Atk', y='Sp. Def', gridsize=50, ax=ax2)
ax2.set_title('Special Attack - Special Defense')
plt.show()


# Same here for Special Attack - Attack and Special Defense - Defense. Naturally it seems to be correlated as well.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
df.plot.hexbin(x='Attack', y='Sp. Atk', gridsize=50, ax=ax1)
ax1.set_title('Special Attack - Attack')
df.plot.hexbin(x='Defense', y='Sp. Def', gridsize=50, ax=ax2)
ax2.set_title('Special Defense - Defense')
plt.show()


# ## Let's try to predict legendary Pokemon with a simple model!
# 
# First let's plot legendary and normal pokemon for attack & defense stats. Even though there is a distinction between both categories, it is not really sharp.

# In[ ]:


legend = df[df['Legendary'] == True]
normal = df[df['Legendary'] != True]
g = sns.JointGrid(x="Attack", y="Defense", data=normal, height=7, ratio=5)
sns.distplot(legend.Attack, ax=g.ax_marg_x)
sns.distplot(legend.Defense, ax=g.ax_marg_y, vertical=True)
sns.distplot(normal.Attack, ax=g.ax_marg_x)
sns.distplot(normal.Defense, ax=g.ax_marg_y, vertical=True)
g.ax_joint.scatter(legend.Attack, legend.Defense, label='Legendary', alpha=0.8)
g.ax_joint.scatter(normal.Attack, normal.Defense, label='Normal', alpha=0.4)
g.ax_joint.legend()
plt.show()


# The same foes with Sp. Attack & Sp. Defense.

# In[ ]:


g = sns.JointGrid(x="Sp. Atk", y="Sp. Def", data=normal, height=7, ratio=5)
sns.distplot(legend['Sp. Atk'], ax=g.ax_marg_x)
sns.distplot(legend['Sp. Def'], ax=g.ax_marg_y, vertical=True)
sns.distplot(normal['Sp. Atk'], ax=g.ax_marg_x)
sns.distplot(normal['Sp. Def'], ax=g.ax_marg_y, vertical=True)
g.ax_joint.scatter(legend['Sp. Atk'], legend['Sp. Def'], label='Legendary', alpha=0.8)
g.ax_joint.scatter(normal['Sp. Atk'], normal['Sp. Def'], label='Normal', alpha=0.4)
g.ax_joint.legend()
plt.show()


# Total seems to be an important feature to dissociate both groups. 

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 7))
legend['Total'].plot.kde(label='Legendary', alpha=0.7)
normal['Total'].plot.kde(label='Normal', alpha=0.7)
ax.legend()
ax.set_title('Total')
plt.show()


# Perhaps it would be interesting to look at the PCA for all the stats. The first PCA component seems to carry most of the information to seperate both groups. It would be interesting to use it as a feature for a model to predict legendary pokemon. However, we can see that the model may predict poorly for several reasons: 
# 
# - Half of the legendary pokemon seems to have fairly regular stats (overlapping with blue points on the graph).
# - There is way more normal pokemon than legendary pokemon.
# 
# The model will surely predict nicely if a pokemon is normal, but may have struggle to predict if a pokemon is legendary.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features = ['HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def']
x = df[features].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
pcs = pca.fit_transform(x)
main_df = pd.DataFrame(data = pcs
             , columns = ['pc1', 'pc2'])
main_df = pd.concat([main_df, df[['Legendary']]], axis = 1)

legend = main_df[main_df['Legendary'] == True]
normal = main_df[main_df['Legendary'] != True]

g = sns.JointGrid(x="pc1", y="pc2", data=normal, height=7, ratio=5)
sns.distplot(legend["pc1"], ax=g.ax_marg_x)
sns.distplot(legend["pc2"], ax=g.ax_marg_y, vertical=True)
sns.distplot(normal["pc1"], ax=g.ax_marg_x)
sns.distplot(normal["pc2"], ax=g.ax_marg_y, vertical=True)
g.ax_joint.scatter(legend["pc1"], legend["pc2"], label='Legendary', alpha=0.8)
g.ax_joint.scatter(normal["pc1"], normal["pc2"], label='Normal', alpha=0.4)
g.ax_joint.legend()
plt.show()


# In[ ]:


pca1 = pca.components_[0]
data = main_df[['pc1', 'Legendary']].sort_values(by='pc1', axis=0)
data.astype(float).plot.scatter(x='pc1', y='Legendary')
plt.title('Legendary')
plt.show()


# Now let's build a simply logistic regression.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from numpy import sort
from sklearn.metrics import confusion_matrix

x = data['pc1'].values.reshape(-1, 1)
y = data['Legendary'].values.astype(float).reshape(800,)
model = LogisticRegression(solver='lbfgs')
model.fit(x, y)

loss = expit(x * model.coef_ + model.intercept_)
fig, ax = plt.subplots(figsize=(15, 7))
ax.scatter(x, y, color='blue', alpha=0.5, label='Real Values')
ax.plot(x, loss, color='red', alpha=0.5, label='Predicted Values')
ax.legend()
plt.show()


# As said earlier, the overall model seems to predict nicely the normal pokemon.

# In[ ]:


y_pred = model.predict(x)
error = (y - y_pred) ** 2
print('  Error: ', error.sum() / error.shape[0], '%')


# However the accuracy for predicting legendary is really low (<50%). Here on the confusion matrix, a lot of legendary pokemon were not classified as legendary.

# In[ ]:


confusion = confusion_matrix(y, y_pred)
sns.heatmap(confusion, annot=True)
plt.show()


# It would be interesting to investigate further parterns for better models: Legendary may have a prefered type, more likely to be double-typed perhaps ? 
