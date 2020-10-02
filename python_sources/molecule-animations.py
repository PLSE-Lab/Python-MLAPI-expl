#!/usr/bin/env python
# coding: utf-8

# # 1JHN Molecule Animations
# In this kernel we display EDA to help us understand how molecular structure predicts `scalar_coupling_constant`. In Kaggle's "Prediction Molecular Properties" competition we are given the structure of molecules as `x,y,z` coordinates in 3D. Then we must predict the `scalar_coupling_constant` between two specific atoms within a molecule.
# 
# We will only be exploring 1JHN couplings. So each plot below shows 1 hydrogen atom at (0,0,0) and 1 nitrogen atom at (1,0,0). We learn from EDA that the additional atoms that connect to the nitrogen affect the value of the `sc_constant`. Typically nitrogen connects with 3 atoms. So after connecting with (the original) hydrogen it most often pairs with 2 more. If it only pairs with one more, it has a small `sc_constant`. If it pairs with another hydrogen it has a small `sc_constant`. If it pairs with larger molecules like carbon or more nitrogen, it has a large `sc_constant`.
# 
# The distance of the bond between the original hydrogen and nitrogen correlates with the `sc_constant`. Also the distances to the additional two atoms that nitrogen connects with help predict `sc_constant`. This is seen in EDA below and confirmed when we build a model using these facts with LGBM.

# In[ ]:


# LOAD LIBRARIES
import numpy as np, pandas as pd, sys, time
from matplotlib.animation import ArtistAnimation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import Image, display
import random, os, gc
import lightgbm as lgb 


# In[ ]:


source_file = '../input/champs-scalar-coupling/structures.csv'
structures_df = pd.read_csv(source_file)
structures_df.head()


# In[ ]:


train_file = '../input/champs-scalar-coupling/train.csv'
train_df = pd.read_csv(train_file)
train_df.head()


# # Display Examples
# Below is the histogram of `scalar_coupling_constant` for bond type `1JHN`. The `sc_constant` ranges from 30 to 70. We will display 10 examples from small `sc_constant` to large `sc_constant` and see if we can detect any patterns. In each animation, we position the hydrogen atom at coordinates (0,0,0). Then we scale the molecule so distance between hydrogen and nitrogen is 1. Next we position the molecule so that nitrogen is at coordinates (1,0,0). Finally we animate the molecule by rotating it around the x-axis. By positioning and scaling all molecules into the same standard position, we can now observe patterns of similaries and differences.

# In[ ]:


BTYPE = '1JHN'
DISPLAY_BONDS = False
train_df = train_df[train_df.type==BTYPE].copy()
plt.figure(figsize=(20,4))
sns.distplot(train_df.scalar_coupling_constant)
plt.title('Bond Type '+BTYPE,fontsize=16)
plt.show()


# In[ ]:


# DIVIDE SCC RANGE INTO NUM DIVS
DIVS=10
# DISPLAY NUM SAMPLES FROM EACH DIV
SAMPLES=1

mn = train_df.scalar_coupling_constant.min()
mx = train_df.scalar_coupling_constant.max()
step = (mx-mn)/DIVS
#print(BTYPE,'sc_constant min =',mn,'max =',mx)
#print('We will display',SAMPLES,'sample(s) from each of',DIVS,'divisions')

# CHOOSE SAMPLES
samples_idx = []; 
for k in range(DIVS):
    ln = len(train_df[ (train_df.scalar_coupling_constant>mn+k*step)&(train_df.scalar_coupling_constant<mn+(k+1)*step)])
    pick = np.min((ln,SAMPLES))
    if pick!=0:
        samples_idx += train_df[ (train_df.scalar_coupling_constant>mn+k*step)&(train_df.scalar_coupling_constant<mn+(k+1)*step)].sample(pick).index.values.tolist()


# In[ ]:


# PUT STRUCTURES IN NUMPY HASH TABLE FOR FAST LOOKOUT
# https://www.kaggle.com/cpmpml/ultra-fast-distance-matrix-computation
np_xyz = structures_df[['x','y','z']].values.astype('float32')
#np_atoms = structures_df['atom'].values
mp = {'H':0,'C':1,'O':2,'N':3,'F':4}
np_atoms = structures_df['atom'].map(mp).values.astype('int8')
ss = structures_df.groupby('molecule_name').size()
ss = ss.cumsum()
ssx = np.zeros(len(ss) + 1, 'int')
ssx[1:] = ss; ssx[0] = 0
np_dict = {}
for k in range(len(ss)):
    np_dict[ss.index[k]] = k

# STANDARDIZE ATOM 1 AND 2 INTO (X,Y) = (0,0) AND (1,0)
def standardizer(mol_name, t1, t2, phi2=0, a_count=0):
    # USES GLOBALS NP_ATOMS, NP_XYZ, SSX, NP_DICT
    atom_df = np_atoms[ ssx[np_dict[mol_name]]:ssx[np_dict[mol_name]+1] ]
    mol_df = np_xyz[ ssx[np_dict[mol_name]]:ssx[np_dict[mol_name]+1], ].copy()
    mol_df -= mol_df[t1,]
    bond_dist = np.sqrt(mol_df[t2,].dot(mol_df[t2,])).astype('float32')
    mol_df /= bond_dist
    
    if mol_df[t2,0]==0: theta = np.sign(mol_df[t2,2])*np.pi/2
    else: theta = np.arctan(mol_df[t2,2]/mol_df[t2,0])
    M = np.array([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]])
    mol_df[:,[0,2]] = (M.dot(mol_df[:,[0,2]].T)).T    
    
    if mol_df[t2,0]==0: phi = np.sign(mol_df[t2,1])*np.pi/2
    else: phi = np.arctan(mol_df[t2,1]/mol_df[t2,0])
    if mol_df[t2,0] < 0: phi += np.pi
    M = np.array([[np.cos(-phi),-np.sin(-phi)],[np.sin(-phi),np.cos(-phi)]])
    mol_df[:,[0,1]] = (M.dot(mol_df[:,[0,1]].T)).T         
      
    M = np.array([[np.cos(-phi2),-np.sin(-phi2)],[np.sin(-phi2),np.cos(-phi2)]])
    mol_df[:,[1,2]] = (M.dot(mol_df[:,[1,2]].T)).T 
            
    # CLOSEST NEIGHBORS TO ATOM_INDEX_1
    neighbor_types = [np.nan]*a_count
    neighbor_dists = [np.nan]*a_count
    if a_count>0:
        dst = np.sum( (mol_df-[1,0,0])*(mol_df-[1,0,0]), axis=1 )
        dst[t1] = 1e3; dst[t2] = 1e3
        idst = np.argsort(dst)
        ln = np.min((len(idst)-2,a_count))
        neighbor_types[:ln] = atom_df[idst[:ln]]
        neighbor_dists[:ln] = dst[idst[:ln]]
        
    return (atom_df, mol_df, bond_dist, neighbor_types, neighbor_dists)


# In[ ]:


def animate(j,title='',display_bonds=DISPLAY_BONDS):
    # CONSTANTS
    atoms = ['H','C','O','N','F']
    colors = ['white','lightgray','red','blue','green']
    sizes = [25, 70, 60, 65, 50]
    lkeys = []
    for k in range(5):
        lkeys.append(mpatches.Patch(facecolor=colors[k], label=atoms[k], edgecolor='black'))
    
    # EDA PLOTS
    mol_name = train_df.loc[j,'molecule_name']
    target_1 = train_df.loc[j,'atom_index_0']
    target_2 = train_df.loc[j,'atom_index_1']
    constant = train_df.loc[j,'scalar_coupling_constant']
    btype = train_df.loc[j,'type']
    frames = []; scl = 1.0
    fig = plt.figure(figsize=(8,5))
    for jj,alpha in enumerate(np.linspace(0,2*np.pi,16)):
        atom_df, mol_df, dis, _, _ = standardizer(mol_name, target_1, target_2, alpha)
        if jj==0: # SCALE CIRCLES IN PLOT
            mxx = np.sqrt(np.max( mol_df[:,1]**2 + mol_df[:,2]**2 ))
            if btype[0]=='2': mxx *= 1.5
            elif btype[0]=='3': mxx *= 2.0
            if mxx<0.1: scl = 200
            else: scl = 87.*(1+0.2*(4-mxx))/mxx
            if display_bonds: scl /= 3.
        # DISPLAY BONDS
        all = []
        if display_bonds:
            ds = mol_df[:,None,:]-mol_df
            dist = np.sqrt(np.einsum('ijk,ijk->ij',ds,ds))
            dist = (dist<1.65/dis).astype(np.int8)
            for k in range(len(mol_df)):
                for kk in range(k+1,len(mol_df)):
                    if dist[k,kk]==1:
                        pp, = plt.plot([mol_df[k,0],mol_df[kk,0]],[mol_df[k,1],mol_df[kk,1]],color='black', zorder=-1 )
                        all.append( pp )
        # ARRANGE ATOMS FROM BACK TO FRONT
        mol_df = np.concatenate([mol_df,atom_df.reshape((-1,1))],axis=1)
        midx = np.argsort(mol_df[:,2],axis=0)
        mol_df = mol_df[midx,]
        atom_df = mol_df[:,3].astype('int8')
        # DISPLAY ATOMS FROM BACK TO FRONT
        for k in range(len(mol_df)):
            all.append( plt.scatter(mol_df[k,0],mol_df[k,1],color=colors[atom_df[k]],edgecolor='black',s=scl*sizes[atom_df[k]]) )
        xlm = plt.xlim()
        plt.legend(handles=lkeys)
        pre = btype
        if title!='': pre = title
        plt.title(pre+', bond '+str(j)+', sc_constant = '+str(np.round(constant,2)),fontsize=16)
        plt.axis('equal')
        frames.append(all)
    # MAKE ANIMATION
    ani = ArtistAnimation(fig, frames)
    ani.save('data'+str(j)+'.gif', writer='imagemagick', fps=5)
    plt.close()
    #print('Bond',j,',Molecule',mol_name)
    with open('data'+str(j)+'.gif','rb') as file:
        display(Image(file.read()))
        
# DISPLAY OUR 10 SAMPLES
for j in samples_idx: animate(j)


# We observe that the atoms connected to nitrogen are associated with `sc_constant`. For the largest `sc_constant`, nitrogen is connected to more nitrogen (either 2N or 1N,1C). For middle `sc_constant`, nitrogen is connected to 2 carbons. For small `sc_constant`, nitrogen is connected to only 1 atom or 2 atoms that include hydrogen. (There is an exception where nitrogen connects to 2 carbons and those carbons connect mainly to hydrogen. That case has low `sc_constant`).
# 
# # Build LGBM Model
# We will build an LGBM model using the types of the closest atoms to the original nitrogen and their distances to nitrogen. From the plots above, we see that these features are predictive.

# In[ ]:


# INITIAL ARRAYS
neighbors = 6
bond_dists = np.zeros((len(train_df),1))
neighbor_types = np.zeros((len(train_df),neighbors))
neighbor_dists = np.zeros((len(train_df),neighbors))
bond_names = []

# COMPUTE FEATURES
for i,k in enumerate(train_df.index):
    bond_names.append(k)
    _, _, a, b, c  = standardizer(train_df.molecule_name[k], 
        train_df.atom_index_0[k], train_df.atom_index_1[k], phi2=0, a_count=neighbors)
    bond_dists[i]=a; neighbor_types[i,]=b; neighbor_dists[i,]=c
    #if i%10000==0: print(i)
    
# TRAINING DATA
X_train = np.concatenate([bond_dists,neighbor_types,neighbor_dists],axis=1)
y_train = train_df['scalar_coupling_constant'].values


# In[ ]:


# COMPETITION METRIC
def mean_log_mae(y_true, y_pred, floor=1e-9):
    mae = np.mean(np.abs(y_true-y_pred))
    return "log_mae", np.log( np.max(mae, floor) ), False

# CREATE 80% TRAIN AND 20% VALIDATION USING UNIQUE MOLECULES
mol_names = train_df.molecule_name.unique()
mols = np.random.choice(mol_names,int(0.8*len(mol_names)),replace=False)
train_df['index'] = np.arange(len(train_df))
idxT = train_df[train_df.molecule_name.isin(mols)]['index'].values
idxV = np.setdiff1d( np.arange(len(X_train)), idxT)

# TRAIN LGBM MODEL
lgbm = lgb.LGBMRegressor(n_estimators=1000, colsample_bytree=0.9, objective='regression', num_leaves=256,
            max_depth=-1, learning_rate=0.05, metric='mae')  
h = lgbm.fit(X_train[idxT,], y_train[idxT], eval_set=[(X_train[idxV,], y_train[idxV])], 
             eval_metric=mean_log_mae, verbose=250)


# # Analysis
# We notice that there is a linear correlation between hydrogen and nitrogen bond distance and `sc_constant` from the plot below. Using just this bond distance, we can predict `sc_constant` with `mae = 3`. That's pretty good. Now using the types of atoms that nitrogen is additionally connected to and their bond distances we see above from LGBM that we can predict `sc_constant` within `mae = 0.5`. That's very good.

# In[ ]:


plt.figure(figsize=(20,5))
plt.scatter(bond_dists,y_train)
plt.title('Distance between N and original H',fontsize=16)
plt.xlabel('Bond distance')
plt.ylabel('Bond coupling')
plt.show()


# In[ ]:


def display_plot(rank,annot=[]):
    atoms = ['H','C','O','N','F']
    colors2 = ['lightgray','darkgray','red','blue','green']
    lkeys = []
    for k in range(5):
        lkeys.append(mpatches.Patch(facecolor=colors2[k], label=atoms[k]))
    postfix=['st','nd','rd','th','th','th','th','th','th','th']
    plt.figure(figsize=(20,5))
    for k in range(5):
        plt.scatter(neighbor_dists[neighbor_types[:,rank]==k,rank],y_train[neighbor_types[:,rank]==k],color=colors2[k])
    plt.title('Distance between N and '+str(rank+1)+postfix[rank]+ 
              ' closest atom with type identified by color',fontsize=16)
    plt.xlabel('Bond distance')
    plt.ylabel('Bond coupling')
    plt.legend(handles=lkeys)
    for k in annot:
        plt.text(k[0],k[1],k[2],fontsize=20)
    plt.show()
    
neigh = 0
display_plot(neigh,[[0.98,50,'A'],[1.55,30,'B'],[1.75,70,'C'],[2,40,'D']])


# The plot above is the distance to the 1st neighbor of nitrogen (excluding the original hydrogen). The color in the plot indicates what type of atom it is. We see that if the closest atom is carbon with bond distance under 1.6, then `sc_constant` is between 30 and 40 (class B). We see that if closest atom is nitrogen, then `sc_constant` is above 60 (class C). If the closest atom is not those 2 cases nor hydrogen with distance 1 (class A above), then it is carbon and it linearly correlates with `sc_constant`. This is class D. From eyeball it looks like `sc_constant = m*x+b` where slope `m= -66` and `b=182`. Below we display classes A and B. We see that the 1st neighbor is hydrogen and carbon respectively.

# In[ ]:


# DISPLAY TYPE A FROM PICTURE ABOVE
neigh = 0 
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==0)&(neighbor_dists[:,neigh]<1.2)).flatten()[5]]
animate(bn,'Class A')


# In[ ]:


# DISPLAY TYPE B FROM PICTURE ABOVE
neigh = 0 
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==1)&(neighbor_dists[:,neigh]<1.6)).flatten()[5]]
animate(bn,'Class B')


# The plot below is the distance to the second closest neighbor to nitrogen (excluding the original hydrogen). We observe that if the second neighbor is hydrogen then `sc_constant` is around 40 (class E). If the second neighbor is nitrogen with distance less than 2 (then this new nitrogen is connected to original nitrogen), then `sc_constant` is around 70 (class F). If the second neighbor is nitrogen with distance greater than 5 then that means that the original nitrogen has only 1 connection and this new nitrogen is 2 bonds away (i.e. not connect directly with first nitrogen) and `sc_constant` is around 30 (class J). You get the idea. Below we plot molecules from classes E, F, I. 

# In[ ]:


neigh = 1
display_plot(neigh,[[0.98,40,'E'],[1.75,70,'F'],[2,50,'G'],[4,33,'H'],[4.8,40,'I'],[5.3,33,'J']])


# In[ ]:


# DISPLAY TYPE E FROM PICTURE ABOVE
neigh = 1
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==0)&(neighbor_dists[:,neigh]<1.5)).flatten()[5]]
animate(bn,'Class E')


# In[ ]:


# DISPLAY TYPE F FROM PICTURE ABOVE
neigh = 1
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==3)&(neighbor_dists[:,neigh]<2)).flatten()[5]]
animate(bn,'Class F')


# In[ ]:


# DISPLAY TYPE G FROM PICTURE ABOVE
neigh = 1
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==2)&(neighbor_dists[:,neigh]>4.5)).flatten()[5]]
animate(bn,'Class I')


# Below are the 3rd closest neighbors to nitrogen. Class K is where carbon is the third closest atom and the distance is less than 2.5. This is only possible if carbon is connected to the nitrogen. Therefore the original nitrogen is connected to at least 4 atoms. This is rare since nitrogen usually connects to 3 atoms. Class K is most likely nitrogen connecting with 3 hydrogens (one is the original) and 1 carbon. The animated plot below confirms this.

# In[ ]:


neigh = 2
display_plot(neigh,[[2.2,42,'K']])


# In[ ]:


# DISPLAY TYPE K FROM PICTURE ABOVE
neigh = 2
bn = bond_names[np.argwhere((neighbor_types[:,neigh]==1)&(neighbor_dists[:,neigh]<2.5)).flatten()[2]]
animate(bn,'Class K')


# Below are plots of the 4th and 5th closest neighbors to nitrogen. Classes L and M are interesting. For a molecule to achieve a distance of 10+ and 15+ with the 4th and 5th closest atom respectively means that the molecule must be long and narrow. These classes of molecules have a small `sc_constant` near 30.

# In[ ]:


display_plot(3,[[10,40,'L']])
display_plot(4,[[15,40,'M']])


# Class M above are molecules whose 5th closest atom is distance 15 away from the original nitrogen. The animation below shows that these molecules are long and narrow.

# In[ ]:


# DISPLAY TYPE M FROM PICTURE ABOVE
neigh = 4
bn = bond_names[np.argwhere( neighbor_dists[:,neigh]>14 ).flatten()[5]]
animate(bn,'Class M')


# # Conclusion
# Viewing EDA helps us understand how models are predicting `sc_constant` in Kaggle's "Prediction Molecular Properties" competition. There are many patterns between the structure of the molecule, the distances between atoms, and `sc_constant`. If a model could understand all the different structures of molecules and have knowledge of distances, it could achieve a very accurate `sc_constant` prediction.
# 
# We also notice that using this naive approach of computing distances to the original nitrogen doesn't work as well for the 3rd, 4th, 5th closest atom etc. At that point, it's hard to tell whether the atoms are chemically connected or not. Using more features and/or GNN (graph neural network) can make better sense of deeper structure further away from the original nitrogen.
