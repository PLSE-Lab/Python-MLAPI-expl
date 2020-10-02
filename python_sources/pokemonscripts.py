# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pkmn = pd.read_csv("../input/Pokemon.csv")
#print(pkmn.shape)
dex = 721

del pkmn['Generation'], pkmn['Legendary']

val = pkmn['Type 1'].unique()

#what type differences are based on primary or secondary type

cnt1 = cnt2 = 0
ind1 = ind2 = 0

avg1 = [0]*len(val)
avg2 = [0]*len(val)
cnt1 = [0]*len(val)
cnt2 = [0]*len(val)
for ind, mytype in enumerate(val):
    for k in list(range(1,len(pkmn['#']))):
        if pkmn['Type 1'][k] == mytype:
            if not pd.isnull(pkmn['Type 2'][k]):
                cnt1[ind] += 1
                avg1[ind] += pkmn["Total"][k]
                #print(pkmn["Name"][k],pkmn['Type 1'][k],pkmn['Type 2'][k])
        
        if pkmn['Type 2'][k] == mytype:
            cnt2[ind] += 1
            avg2[ind] += pkmn["Total"][k]
           
for i in range(0,len(avg1)):
    avg1[i]/=cnt1[i]
    avg2[i]/=cnt2[i]

for i in range(0,len(val)):
    print(val[i], avg1[i], avg2[i], (avg1[i]-avg2[i]))
    


#Most offensive and defensive types 
off_stats = [0]*len(val)
def_stats = [0]*len(val)
for ind, mytype in enumerate(val):
    for k in list(range(1,len(pkmn['#']))):
        if pkmn['Type 1'][k] == mytype or pkmn['Type 2'][k] == mytype:
            #print(mytype)
            #print(pkmn['Name'][k])
            def_stats[ind] += pkmn['HP'][k] + pkmn['Defense'][k] + pkmn['Sp. Def'][k]
            off_stats[ind] += pkmn['Attack'][k] + pkmn['Sp. Atk'][k] + pkmn['Speed'][k]
            

for x in range(0, len(val)):
    off_stats[x] /= (cnt1[x]+cnt2[x])
    def_stats[x] /= (cnt1[x]+cnt2[x])

#print("Offensive and Defensive Stats by Type:")
#for i in range(0,len(val)):
#    print(val[i], off_stats[i], def_stats[i])


#why do normal and electric pokemon have such high averages?

normelec = pkmn.copy()
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

#for k in list(range(1,len(pkmn['#']))):
#        if pkmn['Type 1'][k] not in ['Normal', 'Electric'] and pkmn['Type 2'][k] in ['Normal', 'Electric']:
      
pkmn = pkmn[(pkmn['Type 1'] =="Normal") | (pkmn['Type 1'] =="Electric") | (pkmn['Type 2'] =="Normal") | (pkmn['Type 2'] =="Electric")] 
print(pkmn) 
pkmn_copy = pkmn.copy()
#150 rows, 11 columns for normelec
pkmn_copy[features] = StandardScaler().fit(pkmn[features]).transform(pkmn[features])


X_tsne = TSNE(learning_rate=500, n_components=2).fit_transform(pkmn_copy[features])
X_pca = PCA().fit_transform(pkmn_copy[features])   

plt.figure(figsize=(11, 5))
cmap = plt.get_cmap('nipy_spectral')

plt.subplot(1,2,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cmap(pkmn_copy['Total'] / 2))
plt.title('TSE')
plt.show()
plt.subplot(1,2,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cmap(pkmn_copy['Total'] / 2))
plt.title('PCA');
plt.show()          
            
            
            
            
            
            
            
            
            
            





