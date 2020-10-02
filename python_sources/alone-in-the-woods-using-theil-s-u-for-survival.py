#!/usr/bin/env python
# coding: utf-8

# # Alone in the woods
# You shouldn't have taken that bet. 
# 
# You'll never say that to anyone if you'll ever get out of this forest - but now you should probably focus on that _getting out of the forest_ part.
# 
# Yes, you're an awesome navigator, but so deep in an unknown forest, where you can hardly see the sun, even your great skills are can't give you a clue about where you're headed.
# _"I have a map"_, you said before taking the first step into the woods. True, but you lost it when you almost drowned in that river four days ago. The map didn't said how deep it was.
# 
# But you're here now, and you can still win that bet - if you'll get out here soon. But you're out of food, and you haven't seen a single fruit tree since you day one - and boy, you're a lousy hunter. 
# But hey, there are mushrooms all around. hundreds, if not thousands. You can eat those - if you'd only know which ones you can. Well, can you?
# 
# _"This bet isn't worth this"_ you think. Yes, that's probably right.

# In[ ]:


import math
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/mushrooms.csv')
data.head()


# Let's see how many mushrooms we know about, and what do we know about them:

# In[ ]:


data.shape


# Ok, so we know about 8124 mushrooms, with 22 different attributes (23 includes the `class` column). That's a lot! You can't possibly memorise all this data. Let's try to break it down. What else do we now of our data?

# In[ ]:


data.isnull().values.any()


# Good, no missing data points. What's the varience of each of the features?

# In[ ]:


for feature in data.columns:
    uniq = np.unique(data[feature])
    print('{}: {} distinct values -  {}'.format(feature,len(uniq),uniq))


# Hmmm.. that's interesting - it seems as all known mushrooms have the same `veil-type`. We can drop this feature then, it won't help us.

# In[ ]:


data = data.drop(['veil-type'], axis=1)


# Beore we continue, let's check if there are any duplicates in our data - meaning, do we know of two or more mushrooms with exactly the same features?

# In[ ]:


print('Known mushrooms: {}\nUnique mushrooms: {}'.format(len(data.index),len(data.drop_duplicates().index)))


# No, good. Wait! Are there any mushrooms with the same features but different classes?

# In[ ]:


print('Known mushrooms: {}\nMushrooms with same features: {}'.format(
    len(data.index),len(data.drop_duplicates(subset=data.drop(['class'],axis=1).columns).index)))


# Ok, that's good. That's really good. Being unable to distinguish mushrooms can be deadly. 
# 
# Time to start diving deeper into the data. Let's see if there's a feature that can a better notion about the mushromm's class. More formealy, we want to know if there's a certain feature that maximizes the information about the class. For this, we will use **[Theil's U](https://en.wikipedia.org/wiki/Uncertainty_coefficient)**, also known as the _Uncertainty Coefficient_. Formaly marked as `U(x|y)`, this coefficient provides a value in the range of `[0,1]`, where 0 means that feature `y` provides no information about feature `x`, and 1 means that feature `y` provides full information abpout features `x`'s value.

# In[ ]:


def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


# In[ ]:


theilu = pd.DataFrame(index=['class'],columns=data.columns)
columns = data.columns
for j in range(0,len(columns)):
    u = theil_u(data['class'].tolist(),data[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu,annot=True,fmt='.2f')
plt.show()


# Seems like `odor` can be useful. Lets' take a look at that:

# In[ ]:


sns.set(rc={'figure.figsize':(15,8)})
ax=sns.countplot(x='odor',hue='class',data=data)
for p in ax.patches:
    patch_height = p.get_height()
    if np.isnan(patch_height):
        patch_height = 0
    ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.05, patch_height+10))
plt.show()


# Ok, that's great! We can tell if a mushroom is ediable or not just by smelling it! But what about all those mushrooms with no odor (`odor='n'`)? Let's do the same trick again, but this time only for the data of these mushrooms: 

# In[ ]:


no_odor = data[data['odor'].isin(['n'])]
for j in range(0,len(columns)):
    u = theil_u(no_odor['class'].tolist(),no_odor[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu,annot=True,fmt='.2f')
plt.show()


# Now the `spore-print-color` seems like a helpful feature: 

# In[ ]:


sns.set(rc={'figure.figsize':(15,8)})
ax=sns.countplot(x='spore-print-color',hue='class',data=no_odor)
for p in ax.patches:
    patch_height = p.get_height()
    if np.isnan(patch_height):
        patch_height = 0
    ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.05, patch_height+10))
plt.show()


# Great success! We can tell odorless mushrooms apart by their spore color - unless it's white, then it's still ambiguous.
# 
# Let's see how many mushrooms are we already able telling if they're ediable or not:

# In[ ]:


no_odor_w = no_odor[no_odor['spore-print-color'].isin(['w'])]
(len(data.index) - len(no_odor_w.index)) / len(data.index)


# So we can now determine the class of 92% of the mushrooms we know of. 
# 
# **Let's summarize our findings:**
# * if the mushroom smells like almond (`a`) or anise (`l`), it's ediable.
# * if it has no odor, check its spore color - if it's not green (`r`), it's ediable. If it's white, we're not sure.
# 
# ### A little ML to speed things up:
# We can keep going like this, but 92% percent classified mushroom is enough to help you find some food in the woods - and you only to remeber two things.
# 
# For the remaining 8%, let's use a Decision Tree (well, you are in the forest, aren't you?). The tree will pretty mush do what we did so far - ask the right questions to slice the data in the best way for classifications. Let's see how many branches it needs to get to perfect classification. For this, we'll need to factorize the remaining data and use one-hot encoding.

# In[ ]:


factorized_nw = no_odor_w.copy()
for column in factorized_nw.columns.values:
    f, _ = pd.factorize(factorized_nw[column])
    factorized_nw.loc[:,column] = f
ohe = sp.OneHotEncoder()
X = factorized_nw.drop(['class'],axis=1)
y = factorized_nw['class'].tolist()
ohe.fit(X)
X = ohe.transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

for i in range(1,6):
    tree = DecisionTreeClassifier(max_depth=i, random_state=42)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    print("Max depth: {} - accuracy:".format(i), accuracy_score(y_test, y_pred, normalize=True))


# Seems like we got we wanted, and we only used 50% of the data for training - so we have half of the data as our test set. Let's see some more numbers:

# In[ ]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Amazing! We got it all classified. Come on, now - no more excuses, you got this bet now! **Good luck, and bon appetit!**
