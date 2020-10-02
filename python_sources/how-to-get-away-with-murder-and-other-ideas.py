#!/usr/bin/env python
# coding: utf-8

# **
# 
# How Easy is it to Get Away with Your Crime - Visualization and Dataset Exploration
# ------------------------------------------------------------------------
# 
# **
# Disclaimer: Title is a dig on the TV series How To Get Away With Murder. Written text is meant as satire and in no way represents condonement or advocacy of murder/manslaughter.
# 
# 
# Let's split the data into a few different datasets so that our modifications don't interfere. Start off with some of our imports.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[ ]:


nRecords = 200000
snRecords = 1000
maindf = pd.read_csv('../input/database.csv')


# In[ ]:


#Record ID,Agency Code,Agency Name,Agency Type,City,State,Year,Month,Incident,Crime Type,Crime Solved,Victim Sex,
#Victim Age,Victim Race,Victim Ethnicity,Perpetrator Sex,Perpetrator Age,Perpetrator Race,Perpetrator Ethnicity,
#Relationship,Weapon,Victim Count,Perpetrator Count,Record Source


#This is the main dataframe
maindf.drop(['Year', 'Month','Incident',
 'City','Agency Name','Agency Type','Record Source','Agency Name'], axis=1,inplace=True)

#These two will be our datasets for Tree testing to find feature importance
df = maindf[(maindf['Record ID']<nRecords)]
sdf = df[(maindf['Record ID']<snRecords) & df['Victim Count']>0]


# Let's see a couple of scatterplots for useful info!

# In[ ]:


races = df['Perpetrator Race'].unique()

sns.jointplot(x = 'Perpetrator Count', y = 'Victim Count', data = df)
plt.show()


# The majority of cases seem to have only a single victim, and as the number of perpetrators increases the number of victims seems to drop. 
# Let's look at what factors show correlation with whether a crime is solved or not.

# In[ ]:


sns.countplot(x = df['Perpetrator Race'], hue = df['Crime Solved'], palette = sns.color_palette("Paired",len(races)), data = df)
plt.show()

sns.countplot(x = df['Victim Race'], hue = df['Crime Solved'], palette = sns.color_palette("Paired",len(races)), data = df)
plt.show()

sns.countplot(x = df['Victim Age'], hue = df['Crime Solved'], palette = sns.color_palette("Paired",len(races)), data = df)
plt.show()


# The three graphs above explore how perpetrator race, victim race, and victim age compare with the number of cases solved or not. 
# White perpetrators are almost always caught, and the race of the killer being unknown is a large factor in whether the crime was solved or not. The age graphs peak at around 24 years of age for killers as well as perps - these COULD be crimes of passion, which can be further explored. 
# 

# In[ ]:


#Which weapons claimed the most lives?
sns.swarmplot(x=sdf['Weapon'], y=sdf['Victim Count'].astype(float), data=sdf);
plt.show()

#Does the Victim's sex influence whether the crime is solved or not?
sns.countplot(x = df['Victim Sex'], hue = df['Crime Solved'], palette = sns.color_palette("Paired",len(races)), data = df)
plt.show()

#Do certain weapons happen to be caught more often?
sns.countplot(x = df['Weapon'], hue = df['Crime Solved'], palette = sns.color_palette("Paired",len(races)), data = df)
plt.show()


# Handgun violence is a real issue! And handguns have the most cases with multiple victims, followed by rifles and fires. 
# The ratio of male and female unsolved/solved is similar, so sex doesn't seem to impact whether a case is solvable or not. 

# **If You Want Something Done Right, Do it Yourself**

# In[ ]:


from IPython.display import display

df['Crime Solved'].replace("No",0, inplace = True)
df['Crime Solved'].replace("Yes",1, inplace = True)

df.loc[df['Perpetrator Count'] <=1, 'Perpetrator Count'] = 0
df.loc[df['Perpetrator Count'] >1, 'Perpetrator Count'] = 1

multiple_caught = pd.value_counts((df['Perpetrator Count']*(df['Crime Solved'])), sort = False)[1]
one_escape = pd.value_counts((df['Perpetrator Count'] | df['Crime Solved']),sort = False)[0]
one_caught = pd.value_counts((df['Perpetrator Count'] < df['Crime Solved']), sort = False)[True]
multiple_escape = pd.value_counts((df['Perpetrator Count'] > df['Crime Solved']), sort = False)[True]

res = np.matrix([[one_caught, multiple_caught],[one_escape,multiple_escape]])
print(pd.DataFrame(res))


# There were roughly 194000 crimes committed by a single perpetrator, 141000 of which were solved and 54000 were not. 5100 crimes were committed by multiple perpetrators, and 11 of those were unsolved.
# 
# 

# In[ ]:


print("Percentage of unsolved cases with one perp: %f \nPercentage of unsolved cases with multiple perps: " %((one_escape/(one_caught+one_escape))),(multiple_escape/(multiple_caught+multiple_escape)))


# In[ ]:


Moral: You want something done well, do it yourself. Or get trustworthy buddies who won't squeal.


# ***Most Important Features***
# 
# Let's use a random forest to analyze which features are most useful in determining whether a crime is solved or not.

# In[ ]:


treedf = maindf[(maindf['Record ID']>=nRecords)&(maindf['Record ID']<2*nRecords)]
n_estimators = 10
clf = RandomForestClassifier(n_estimators)


#Preparing training set
dfcopy = df.copy(deep = True)
random_y = df['Crime Solved']
dfcopy.drop('Crime Solved', axis = 1, inplace = True)


for z in dfcopy.columns:
	if isinstance(z[0],str):
		arr = dfcopy[z].unique()		
		le = preprocessing.LabelEncoder()
		le.fit(arr)
		dfcopy[z] = le.transform(dfcopy[z])

clf.fit(dfcopy, random_y)

scores = clf.score(dfcopy, random_y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking for first set:")

for f in range(dfcopy.shape[1]):
    print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], dfcopy.columns[indices[f]]))

#Prepare test data
treedfcopy = treedf.copy(deep = True)
test_y = treedf['Crime Solved']
treedfcopy.drop('Crime Solved', axis = 1, inplace = True)

for z in treedfcopy.columns:
	if isinstance(z[0],str):
		arr = treedfcopy[z].unique()		
		le = preprocessing.LabelEncoder()
		le.fit(arr)
		treedfcopy[z] = le.transform(treedfcopy[z])

clf.fit(treedfcopy, test_y)
scores = clf.score(treedfcopy, test_y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature ranking for second set:")

for f in range(treedfcopy.shape[1]):
    print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], treedfcopy.columns[indices[f]]))


# Most valuable features above 15%: Perpetrator Sex, Perpetrator Age, Perpetrator Race.

# Let's see some structure with TSNE/PCA!

# In[ ]:


for z in sdf.columns:
	if isinstance(z[0],str):
		arr = sdf[z].unique()		
		le = preprocessing.LabelEncoder()
		le.fit(arr)
		sdf[z] = le.transform(sdf[z])


X_tsne = TSNE(perplexity = 45,learning_rate= 900, n_components=2).fit_transform(sdf)
X_pca = PCA(n_components = 2).fit_transform(sdf)

plt.figure(figsize=(11, 5))
cmap = plt.get_cmap('nipy_spectral')

plt.subplot(1,2,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('TSNE')
plt.subplot(1,2,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA');
plt.show()


# In[ ]:


TSNE gives quite some structure!


# In[ ]:


num_clusters = 3
kmeans_tsne = KMeans(n_clusters=num_clusters).fit(X_tsne)
kmeans_pca = KMeans(n_clusters=num_clusters).fit(X_pca)

plt.figure(figsize=(12, 5))
cmap = plt.get_cmap('nipy_spectral')

plt.subplot(1,2,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cmap(kmeans_tsne.labels_ / num_clusters))
plt.title('kmeans-TSNE')
plt.subplot(1,2,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cmap(kmeans_pca.labels_ / num_clusters))
plt.title('kmeans-PCA');
plt.show()

