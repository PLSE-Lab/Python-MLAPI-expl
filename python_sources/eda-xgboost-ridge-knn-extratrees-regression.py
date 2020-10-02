#!/usr/bin/env python
# coding: utf-8

# THIS NOTEBOOK CONTAINS 4 DIFFERENT REGRESSION MODELS AND THEIR AVERAGE AS OUTPUT.
# I COMMENTED SOME PARTS DUE TO 1-HOUR CONSTRAINT.
# 
# 

# In[ ]:


#Importing libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import scipy as sci
import seaborn as sns
import matplotlib.pyplot as plt
#For notebook format
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


import missingno as msno
msno.bar(train,sort=True,figsize=(4,2))


# In[ ]:


msno.bar(test,sort=True,figsize=(4,2))


# In[ ]:


train['formation_energy_ev_natom'] = np.log(train['formation_energy_ev_natom'] +1)
train['bandgap_energy_ev'] = np.log(train['bandgap_energy_ev'] + 1 )


# In[ ]:


len(set(train['number_of_total_atoms']))


# In[ ]:


for i in list(train.columns.values):
    print(len(set(train[i])))


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train['spacegroup'].value_counts(normalize=True).plot(kind='bar',figsize=(6,3))


# In[ ]:


test['spacegroup'].value_counts(normalize=True).plot(kind='bar',figsize=(6,3))


# In[ ]:


train.groupby('spacegroup')[['formation_energy_ev_natom','bandgap_energy_ev']].mean().plot(kind='bar',figsize=(6,3))
plt.title('Mean of spacegroup')


# In[ ]:


#Discriminating train and test
train['train_or_not'] = 1
test['train_or_not'] = 0


# In[ ]:


data = pd.concat([train,test],ignore_index=True)


# In[ ]:


import math


# In[ ]:


def one_to_two(x):
    a = x['lattice_vector_1_ang']
    b = x['lattice_vector_2_ang']
    c = x['lattice_angle_gamma_degree']
    return np.sqrt( a**2 + b**2 - 2*a*b*math.cos(math.radians(c)))


# In[ ]:


data['one_to_two'] = data.apply(one_to_two,axis=1)


# In[ ]:


def one_to_three(x):
    a = x['lattice_vector_1_ang']
    b = x['lattice_vector_3_ang']
    c = x['lattice_angle_beta_degree']
    return np.sqrt( a**2 + b**2 - 2*a*b*math.cos(math.radians(c)))


# In[ ]:


data['one_to_three'] = data.apply(one_to_three,axis=1)


# In[ ]:


def two_to_three(x):
    a = x['lattice_vector_2_ang']
    b = x['lattice_vector_3_ang']
    c = x['lattice_angle_alpha_degree']
    return np.sqrt(a**2 + b**2 - 2*a*b*math.cos(math.radians(c)))


# In[ ]:


data['two_to_three'] = data.apply(two_to_three,axis=1)


# In[ ]:


conti = pd.crosstab(data['spacegroup'], data['number_of_total_atoms'])


# In[ ]:


conti


# In[ ]:


#Number of atoms = 80 and the counts over spacegroups
((data[(data['train_or_not'] ==1)&(data['number_of_total_atoms'] ==80.0)].groupby('spacegroup')['bandgap_energy_ev'].mean())/((data[(data['train_or_not'] ==1)&(data['number_of_total_atoms'] ==80.0)].groupby('spacegroup')['formation_energy_ev_natom'].mean()))).plot(kind='barh')


# In[ ]:


data['bucket1_sg'] = 0
data['bucket2_sg'] = 0
data['bucket1_sg'][((data['spacegroup'] == 12) | (data['spacegroup'] == 206)) & (data['number_of_total_atoms'] ==80.0)] = 1
data['bucket2_sg'][(data['spacegroup'] == 194) & (data['number_of_total_atoms'] ==80.0)] = 1


# In[ ]:


"""
spacegroup_no_of_atoms = [[x,y] for x in list(set(data['spacegroup'])) for y in list(set(data['number_of_total_atoms']))]
for i in range(len(spacegroup_no_of_atoms)):
    k = spacegroup_no_of_atoms[i][0]
    l = spacegroup_no_of_atoms[i][1]
    if len(data[(data['spacegroup'] ==k) & (data['number_of_total_atoms'] == l)]) >0:
        data['spacegroup_'+str(k)+'_'+'nota_'+str(l)] = 0
        data['spacegroup_'+str(k)+'_'+'nota_'+str(l)][(data['spacegroup'] ==k) & (data['number_of_total_atoms'] ==l)] =1
"""    


# In[ ]:


data.head()


# In[ ]:


data.plot(kind="scatter", x="lattice_vector_1_ang", y="lattice_vector_2_ang",figsize=(6,3))


# In[ ]:


data.plot(kind="scatter", x="lattice_vector_1_ang", y="lattice_vector_3_ang",figsize=(6,3))


# In[ ]:


data.plot(kind="scatter", x="lattice_vector_2_ang", y="lattice_vector_3_ang",figsize=(6,3))


# In[ ]:


#XYZ coordinates and Number_of_total_atoms
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(19945133)
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['lattice_vector_1_ang'], data['lattice_vector_2_ang'], data['lattice_vector_3_ang'], c=data['number_of_total_atoms'], marker='.')
ax.set_xlabel('lattice_vector_1_ang')
ax.set_ylabel('lattice_vector_2_ang')
ax.set_zlabel('lattice_vector_3_ang')
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
lattice_vector_123 = np.array(data[['lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang']])
kmeans = KMeans(n_clusters=10, random_state=51).fit(lattice_vector_123)


# In[ ]:


clusterings = pd.Series(kmeans.labels_)
clusterings.name = 'clusters_of_lattices'
data = pd.concat([data,clusterings],axis=1)


# In[ ]:


df1 = pd.get_dummies(data['clusters_of_lattices'])
df1.columns = ['clusters_of_lattices_'+ str(r) for r in list(df1.columns.values)]
data = pd.concat([data,df1],axis=1)


# In[ ]:


data.head()


# In[ ]:


data[data['train_or_not'] ==1 ].groupby('clusters_of_lattices')[['bandgap_energy_ev','formation_energy_ev_natom']].mean().plot(kind='bar')


# In[ ]:


#XYZ coordinates and Number_of_total_atoms
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(19945133)
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['lattice_vector_1_ang'], data['lattice_vector_2_ang'], data['lattice_vector_3_ang'], c=data['clusters_of_lattices'], marker='.',)
ax.set_xlabel('lattice_vector_1_ang')
ax.set_ylabel('lattice_vector_2_ang')
ax.set_zlabel('lattice_vector_3_ang')
plt.title('clustered version of lattices')
plt.show()


# In[ ]:


data.plot(kind="scatter", x="one_to_two", y="one_to_three",figsize=(6,3))


# In[ ]:


data.plot(kind="scatter", x="one_to_two", y="two_to_three",figsize=(6,3))


# In[ ]:


data.plot(kind="scatter", x="one_to_three", y="two_to_three",figsize=(6,3),marker='.')


# In[ ]:


print(data.columns.values)


# # CORRELATION

# In[ ]:


corr_list = ['percent_atom_al','percent_atom_ga','percent_atom_in','bandgap_energy_ev','formation_energy_ev_natom']


# In[ ]:


corr = data[data['train_or_not'] ==1][corr_list].corr()
f,ax = plt.subplots(figsize=(12,9))

#Draw the heatmap using seaborn
sns.heatmap(corr, cmap='winter_r', annot=True)


# In[ ]:


data[data['train_or_not'] ==1]['bandgap_energy_ev'].hist(bins = 100,color = 'green')


# In[ ]:


data[data['train_or_not'] ==1]['formation_energy_ev_natom'].hist(bins = 100,color='red')


# In[ ]:


data.head()


# In[ ]:


def area_calculator(x):
    a = x['one_to_two']
    b = x['one_to_three']
    c = x['two_to_three']
    p = (a+b+c)/2
    return np.sqrt(p*(p-a)*(p-b)*(p-c))


# In[ ]:


data['area'] = data.apply(area_calculator,axis=1)


# In[ ]:


data.head()


# In[ ]:


data[data['train_or_not'] ==1][['area','formation_energy_ev_natom','bandgap_energy_ev']].corr()


# In[ ]:


data[data['train_or_not'] ==1].plot(kind = "scatter",x='area',y ='formation_energy_ev_natom',marker='.',figsize=(10,5))


# In[ ]:


def area_bucket(x):
    if (x>10) & (x<65):
        return 1
    elif (x>65) & (x<95):
        return 2
    elif (x>95) & (x<120):
        return 3
    else:
        return 4


# In[ ]:


data['area_bucket'] = data['area'].map(lambda x: area_bucket(x))


# In[ ]:


data.groupby('area_bucket')['formation_energy_ev_natom'].mean()


# In[ ]:


data['area_bucket_3'] = data['area_bucket'].map(lambda x: 1 if x==3 else 0)
data['area_bucket_4'] = data['area_bucket'].map(lambda x: 1 if x==4 else 0)


# In[ ]:


del data['area_bucket']


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
colors = ['b', 'c', 'y', 'm', 'r','g','k','pink','orange','purple']

c0 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 0)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 0)]['formation_energy_ev_natom'], marker='x', color=colors[0])
c1 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 1)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 1)]['formation_energy_ev_natom'], marker='o', color=colors[1])
c2 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 2)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 2)]['formation_energy_ev_natom'], marker='.', color=colors[2])
c3 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 3)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 3)]['formation_energy_ev_natom'], marker='^', color=colors[3])
c4 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 4)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 4)]['formation_energy_ev_natom'], marker='+', color=colors[4])
c5 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 5)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 5)]['formation_energy_ev_natom'], marker='v', color=colors[5])
c6 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 6)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 6)]['formation_energy_ev_natom'], marker='_', color=colors[6])
c7 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 7)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 7)]['formation_energy_ev_natom'], marker='*', color=colors[7])
c8 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 8)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 8)]['formation_energy_ev_natom'], marker='s', color=colors[8])
c9 = plt.scatter(data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 9)]['bandgap_energy_ev'], data[(data['train_or_not'] ==1)&(data['clusters_of_lattices'] == 9)]['formation_energy_ev_natom'], marker='d', color=colors[9])

plt
plt.xlabel("bandgap_energy_ev")
plt.ylabel("formation_energy_ev_natom")
plt.legend((c0, c1, c2, c3, c4, c5, c6,c7,c8,c9),
           ('cluster0','cluster1','cluster2','cluster3','cluster4','cluster5','cluster6','cluster7','cluster8','cluster9'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=10)

plt.show()


# In[ ]:


rcParams['figure.figsize'] = 10, 5

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['lattice_angle_alpha_degree'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['lattice_angle_alpha_degree'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Lattice angle alpha degree vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('energy')
ax[0].set_xlabel('lattice alpha degree')
ax[1].set_ylabel('energy')
ax[1].set_xlabel('lattice alpha degree')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


len(data[(data['lattice_angle_alpha_degree']<92.5) & (data['lattice_angle_alpha_degree']>89.5)])


# In[ ]:


rcParams['figure.figsize'] = 10, 5

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['lattice_angle_beta_degree'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['lattice_angle_beta_degree'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Lattice angle beta degree vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('energy')
ax[0].set_xlabel('Lattice angle beta degree')
ax[1].set_ylabel('energy')
ax[1].set_xlabel('Lattice angle beta degree')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


print(data[(data['train_or_not'] ==1) & (data['lattice_angle_beta_degree'] > 102)]['formation_energy_ev_natom'].mean())
print(data[(data['train_or_not'] ==1) & (data['lattice_angle_beta_degree'] < 102)]['formation_energy_ev_natom'].mean())


# In[ ]:


data['beta_bigger_102'] = data['lattice_angle_beta_degree'].map(lambda x: 1 if x >102 else 0)


# In[ ]:


print(data[(data['train_or_not'] ==1) & (data['lattice_angle_beta_degree'] > 102)]['bandgap_energy_ev'].mean())
print(data[(data['train_or_not'] ==1) & (data['lattice_angle_beta_degree'] < 102)]['bandgap_energy_ev'].mean())


# In[ ]:


len(data[(data['lattice_angle_beta_degree']<90.5) & (data['lattice_angle_beta_degree']>89.5)])


# In[ ]:


rcParams['figure.figsize'] = 8, 4

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['lattice_angle_gamma_degree'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['lattice_angle_gamma_degree'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Lattice angle gamma degree vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('energy')
ax[0].set_xlabel('degree')
ax[1].set_ylabel('energy')
ax[1].set_xlabel('degree')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


data['atomic_density'] = data['number_of_total_atoms']/(data['lattice_vector_1_ang']*data['lattice_vector_2_ang']*data['lattice_vector_3_ang'])


# In[ ]:


rcParams['figure.figsize'] = 8, 4

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['number_of_total_atoms'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['number_of_total_atoms'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Lattice angle gamma degree vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('energy')
ax[0].set_xlabel('degree')
ax[1].set_ylabel('energy')
ax[1].set_xlabel('degree')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


data[(data['number_of_total_atoms']==10) & (data['formation_energy_ev_natom']>0.3)]


# In[ ]:


data = data.drop([1235,1983])


# In[ ]:


data.shape


# # number_of_total_atoms

# In[ ]:


data[data['train_or_not'] ==1].groupby('number_of_total_atoms')[['formation_energy_ev_natom','bandgap_energy_ev']].mean().plot(kind='barh',figsize=(10,5),legend='reverse')
plt.title('Number of total atoms')


# In[ ]:


data['number_of_total_atoms'].value_counts().sort_values(ascending=True).plot(kind='bar')


# In[ ]:


df1 = pd.get_dummies(data['number_of_total_atoms'])
df1.columns = ['nota_'+ str(x) for x in list(df1.columns.values)]


# In[ ]:


data = pd.concat([data,df1],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


#AL vs Energy
rcParams['figure.figsize'] = 12, 6

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['percent_atom_al'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['percent_atom_al'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Percent of AL vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('Formation energy')
ax[0].set_xlabel('AL percent')
ax[1].set_ylabel('Bandgap energy')
ax[1].set_xlabel('AL percent')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


#The more the AL percent , The more the BandGap Energy


# In[ ]:


print("number of unique values in al percent = ", len(set(data['percent_atom_al'])))


# In[ ]:


print(sorted(list(set(data['percent_atom_al']))))


# In[ ]:


#GA vs Energy
rcParams['figure.figsize'] = 12, 6

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['percent_atom_ga'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['percent_atom_ga'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Percent of GA vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('Formation energy')
ax[0].set_xlabel('GA percent')
ax[1].set_ylabel('Bandgap energy')
ax[1].set_xlabel('GA percent')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


print("number of unique values in ga percent = ", len(set(data['percent_atom_ga'])))


# In[ ]:


print(sorted(list(set(data['percent_atom_ga']))))


# In[ ]:


#IN vs Energy
rcParams['figure.figsize'] = 12, 6

fig, ax = plt.subplots(nrows = 2,ncols=1, sharex=True, sharey=True,)
ax[0].scatter(data[data['train_or_not'] ==1]['percent_atom_in'].values, data[data['train_or_not'] ==1]['formation_energy_ev_natom'].values,
              color='blue', s=3, label='formation', alpha=1)
ax[1].scatter(data[data['train_or_not'] ==1]['percent_atom_in'].values, data[data['train_or_not'] ==1]['bandgap_energy_ev'].values,
              color='green', s=3, label='bandgap', alpha=1)
fig.suptitle('Percent of In vs formation and bandgap energy')
ax[0].legend(loc=0)
ax[0].set_ylabel('Formation energy')
ax[0].set_xlabel('In percent')
ax[1].set_ylabel('Bandgap energy')
ax[1].set_xlabel('In percent')
ax[1].legend(loc=0)

plt.show()


# In[ ]:


print("number of unique values in ga percent = ", len(set(data['percent_atom_in'])))


# In[ ]:


print(sorted(list(set(data['percent_atom_in']))))


# In[ ]:


for j in list(set(data['spacegroup'])):
    rcParams['figure.figsize'] = 4, 2
    plt.scatter(data[(data['train_or_not'] ==1)&(data['spacegroup'] ==j)]['formation_energy_ev_natom'] , data[(data['train_or_not'] ==1)&(data['spacegroup'] ==j)]['bandgap_energy_ev'],marker = '.',c='r',s=4)
    plt.title("scatter plot of spacegroup = " + str(j))
    plt.xlabel("formation energy")
    plt.ylabel("bandgap energy")
    plt.show()


# In[ ]:


#Getting rid of outliers
data[(data['train_or_not'] ==1) & (data['spacegroup'] ==206) & (data['formation_energy_ev_natom']>0.3)]


# In[ ]:


data = data.drop([307,1378])


# In[ ]:


data.shape


# In[ ]:


#Total weight of Al = 27,GA = 70, IN = 115
#Proton number of AL =13, GA =31, IN =49
#Total weight
data['weight_number'] = data['number_of_total_atoms']*(data['percent_atom_al']*27 +data['percent_atom_ga']*70 + data['percent_atom_in']*115)
#PROTON nUMBER
data['proton_number'] = data['number_of_total_atoms']*(data['percent_atom_al']*13 +data['percent_atom_ga']*31 + data['percent_atom_in']*49)


# In[ ]:


#Weighted Density
data['weighted_density'] = data['weight_number']/(data['lattice_vector_1_ang']* data['lattice_vector_2_ang'] * data['lattice_vector_3_ang'])


# In[ ]:


df1 = pd.get_dummies(data['spacegroup'])
df1.columns = ['spacegroup_'+ str(r) for r in list(df1.columns.values)]


# In[ ]:


data = pd.concat([data,df1],axis=1)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data = data.reset_index(drop=True)


# In[ ]:


train = data[data['train_or_not'] ==1]
test = data[data['train_or_not'] ==0]


# In[ ]:


train_id = train['id']
test_id = test['id']


# In[ ]:


test_id = test_id.reset_index(drop=True)


# In[ ]:


test_id.head()


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


y1 = train['bandgap_energy_ev']
y2 = train['formation_energy_ev_natom']


# In[ ]:


del train['formation_energy_ev_natom'],train['bandgap_energy_ev'],train['train_or_not'],train['spacegroup'],train['id']


# In[ ]:


del test['train_or_not'],test['formation_energy_ev_natom'],test['bandgap_energy_ev'],test['spacegroup'],test['id']


# In[ ]:


del train['clusters_of_lattices'],test['clusters_of_lattices']


# In[ ]:


train.head()


# # XGBOOST FOR Y1- BANDGAP

# In[ ]:


from sklearn.model_selection import KFold
from xgboost import XGBRegressor


# In[ ]:


X = train.values
y1 = y1.values
y2 = y2.values


# In[ ]:


kf = KFold(n_splits=5,random_state= 33)
print(kf)


# In[ ]:


md,lr,ne,mcw,csbt,csbl,l2,ss = [4,6,8],[0.03,0.05,0.07],[100,125],[10,15],[0.7,0.8],[0.8,0.9],[5],[0.8]
params = [[x,y,z,t,v,u,k,l] for x in md for y in lr for z in ne for t in mcw for v in csbt for u in csbl for k in l2 for l in ss]
print(len(params))


# In[ ]:


def rmsle(a,b):
    return np.sqrt(np.mean(np.square( np.log( (np.exp(a)-1) + 1 ) - np.log((np.exp(b)-1)+1) )))


# In[ ]:


params_dict = {}


# In[ ]:


"""
%%time
for i in range(len(params)):
    error_rate = []
    model1 = XGBRegressor(eval_metric = 'rmse',max_depth = params[i][0],learning_rate=params[i][1],n_estimators=params[i][2],min_child_weight=params[i][3],colsample_bytree=params[i][4],colsample_bylevel=params[i][5],reg_lambda = params[i][6],subsample=params[i][7],n_jobs=4)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y1[train_index], y1[test_index]
        model1.fit(X_train,y_train)
        y_pred = model1.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 100 ==0:
        print(i)"""


# In[ ]:


"""params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:10]"""


# In[ ]:


model1 = XGBRegressor(eval_metric = 'rmse',max_depth = 4,learning_rate=0.07,n_estimators=125,min_child_weight=15,subsample=0.8,colsample_bytree=0.8,colsample_bylevel=0.8,reg_lambda =5, n_jobs=4)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model1.fit(X,y1)')


# In[ ]:


#Training error
print("training error = ", round(rmsle(model1.predict(X),y1),5))


# In[ ]:


xgboost_y1_train = model1.predict(X)
xgboost_y1_test = model1.predict(test.values)
xgboost_y1_train = pd.Series(xgboost_y1_train)
xgboost_y1_test = pd.Series(xgboost_y1_test)


# *#XGBOOST FOR Y2*****

# In[ ]:


md,lr,ne,mcw,csbt,csbl,l2,ss = [4,6,8],[0.03,0.05,0.07],[100,125],[10,15],[0.7,0.8],[0.8,0.9],[5],[0.6,0.8]
params = [[x,y,z,t,v,u,k,l] for x in md for y in lr for z in ne for t in mcw for v in csbt for u in csbl for k in l2 for l in ss]
print(len(params))


# In[ ]:


params_dict2 = {}


# In[ ]:


"""%%time
for i in range(len(params)):
    error_rate = []
    model2 = XGBRegressor(eval_metric = 'rmse',max_depth = params[i][0],learning_rate=params[i][1],n_estimators=params[i][2],min_child_weight=params[i][3],colsample_bytree=params[i][4],colsample_bylevel=params[i][5],reg_lambda = params[i][6],subsample=params[i][7],n_jobs=-1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        model2.fit(X_train,y_train)
        y_pred = model2.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict2[str(params[i])] = round(np.mean(error_rate),5)
    if i % 100 ==0:
        print(i)"""


# In[ ]:


"""params_df2 = pd.Series(params_dict2)
print(len(params_dict2))
params_df2 = params_df2.sort_values(ascending=True)
params_df2[:10]"""


# In[ ]:


model2 = XGBRegressor(eval_metric = 'rmse',max_depth = 6,learning_rate=0.07,n_estimators=100,min_child_weight=15,subsample=0.8,colsample_bytree=0.7,colsample_bylevel=0.9,reg_lambda =5, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model2.fit(X,y2)')


# In[ ]:


#Training error
print("training error for y2 = ", round(rmsle(model2.predict(X),y2),5))


# In[ ]:


xgboost_y2_train = model2.predict(X)
xgboost_y2_test = model2.predict(test.values)
xgboost_y2_train = pd.Series(xgboost_y2_train)
xgboost_y2_test = pd.Series(xgboost_y2_test)


# In[ ]:


len(xgboost_y1_train),len(xgboost_y1_test),len(xgboost_y2_train),len(xgboost_y2_test)


# # RIDGE REGRESSION- y1(bandgap)

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


alphas,solvers = [0.0001,0.001,0.01,0.1,1,5,10],['auto', 'svd', 'cholesky', 'lsqr']
params = [[x,y] for x in alphas for y in solvers]
print(len(params))


# In[ ]:


params_dict = {}


# In[ ]:


"""%%time
for i in range(len(params)):
    error_rate = []
    model1 = Ridge(alpha = params[i][0],solver=params[i][1])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y1[train_index], y1[test_index]
        model1.fit(X_train,y_train)
        y_pred = model1.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


"""params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:10]"""


# In[ ]:


model1 = Ridge(alpha =0.0001 ,solver ='auto')
model1.fit(X,y1)
#Training error
print("training error for y1 = ", round(rmsle(model1.predict(X),y1),5))


# In[ ]:


ridge_y1_train = model1.predict(X)
ridge_y1_test = model1.predict(test.values)
ridge_y1_train = pd.Series(ridge_y1_train)
ridge_y1_test = pd.Series(ridge_y1_test)


# # RIDGE REGRESSION FOR Y2= FORMATION

# In[ ]:


alphas,solvers = [0.0001,0.001,0.01,0.1,1,5,10],['auto', 'svd', 'cholesky', 'lsqr']
params2 = [[x,y] for x in alphas for y in solvers]
print(len(params2))


# In[ ]:


params_dict2 = {}


# In[ ]:


"""
%%time
for i in range(len(params2)):
    error_rate = []
    model2 = Ridge(alpha = params2[i][0],solver=params2[i][1])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        model2.fit(X_train,y_train)
        y_pred = model2.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict2[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


params_df2 = pd.Series(params_dict2)
print(len(params_dict2))
params_df2 = params_df2.sort_values(ascending=True)
params_df2[:10]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model2 = Ridge(alpha =0.001 ,solver =\'auto\')\nmodel2.fit(X,y2)\n#Training error\nprint("training error for y1 = ", round(rmsle(model2.predict(X),y2),5))\nridge_y2_train = model2.predict(X)\nridge_y2_test = model2.predict(test.values)\nridge_y2_train = pd.Series(ridge_y2_train)\nridge_y2_test = pd.Series(ridge_y2_test)')


# # EXTRA TREEES REGRESSOR for y1

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


mf,n_e,mss = ["auto","sqrt","log2"],[50,100,150,200],[2,5,10,20]
params = [[x,y,z] for x in mf for y in n_e for z in mss]
print(len(params))
params_dict = {}


# In[ ]:


"""%%time
for i in range(len(params)):
    error_rate = []
    model1 = ExtraTreesRegressor(max_features = params[i][0],n_estimators =params[i][1],min_samples_split = params[i][2],n_jobs=4,random_state=51)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y1[train_index], y1[test_index]
        model1.fit(X_train,y_train)
        y_pred = model1.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


"""params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:10]"""


# In[ ]:


model1 = ExtraTreesRegressor(max_features ="auto" ,n_estimators =100,min_samples_split=20)
model1.fit(X,y1)
#Training error
print("training error for y1 = ", round(rmsle(model1.predict(X),y1),5))


# In[ ]:


et_y1_train = model1.predict(X)
et_y1_test = model1.predict(test.values)
et_y1_train = pd.Series(et_y1_train)
et_y1_test = pd.Series(et_y1_test)


# # EXTRA TREES REGRESSOR FOR Y2

# In[ ]:


mf,n_e,mss = ["auto","sqrt","log2"],[50,100,150,200],[2,5,10,20]
params2 = [[x,y,z] for x in mf for y in n_e for z in mss]
print(len(params2))
params_dict2 = {}


# In[ ]:


"""
%%time
for i in range(len(params2)):
    error_rate = []
    model2 = ExtraTreesRegressor(max_features = params2[i][0],n_estimators =params2[i][1],min_samples_split = params2[i][2],n_jobs=4,random_state=51)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        model2.fit(X_train,y_train)
        y_pred = model2.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict2[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


"""
params_df2 = pd.Series(params_dict2)
print(len(params_dict2))
params_df2 = params_df2.sort_values(ascending=True)
params_df2[:10]
"""


# In[ ]:


model2 = ExtraTreesRegressor(max_features ="auto" ,n_estimators =150,min_samples_split=20)
model2.fit(X,y2)
#Training error
print("training error for y2 = ", round(rmsle(model2.predict(X),y2),5))


# In[ ]:


et_y2_train = model2.predict(X)
et_y2_test = model2.predict(test.values)
et_y2_train = pd.Series(et_y2_train)
et_y2_test = pd.Series(et_y2_test)


# # KNN REGRESSION for y1

# In[ ]:


#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.fit_transform(test.values)


# In[ ]:


print(X_scaled[:1])


# In[ ]:


#Parameters for KNN
from sklearn.neighbors import KNeighborsRegressor
n_n,we,dis,alg = [3,5,7,9,11],['uniform','distance'],["minkowski","manhattan","euclidean","chebyshev"], ['auto', 'ball_tree', 'kd_tree', 'brute']
params = [[x,y,z,t] for x in n_n for y in we for z in dis for t in alg]
print(len(params))
params_dict = {}


# In[ ]:


"""
%%time
for i in range(len(params)):
    error_rate = []
    model1 = KNeighborsRegressor(n_neighbors = params[i][0],weights =params[i][1],metric = params[i][2],algorithm = params[i][3],n_jobs=4)
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y1[train_index], y1[test_index]
        model1.fit(X_train,y_train)
        y_pred = model1.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


"""params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:10]"""


# In[ ]:


model1 = KNeighborsRegressor(n_neighbors = 11,weights ='uniform',metric = 'manhattan',algorithm = 'brute',n_jobs=4)
model1.fit(X_scaled,y1)
#Training error
print("training error for y1 = ", round(rmsle(model1.predict(X_scaled),y1),5))


# In[ ]:


knn_y1_train = model1.predict(X_scaled)
knn_y1_test = model1.predict(test_scaled)
knn_y1_train = pd.Series(knn_y1_train)
knn_y1_test = pd.Series(knn_y1_test)


# # KNN regression for y2

# In[ ]:


#Parameters for KNN
n_n,we,dis,alg = [3,5,7,9,11],['uniform','distance'],["minkowski","manhattan","euclidean","chebyshev"], ['auto', 'ball_tree', 'kd_tree', 'brute']
params2 = [[x,y,z,t] for x in n_n for y in we for z in dis for t in alg]
print(len(params2))
params_dict2 = {}


# In[ ]:


"""%%time
for i in range(len(params2)):
    error_rate = []
    model2 = KNeighborsRegressor(n_neighbors = params2[i][0],weights =params2[i][1],metric = params2[i][2],algorithm = params2[i][3],n_jobs=4)
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        model2.fit(X_train,y_train)
        y_pred = model2.predict(X_test)
        rmsle_calculated = rmsle(y_pred,y_test)
        error_rate.append(rmsle_calculated) 
    params_dict2[str(params[i])] = round(np.mean(error_rate),5)
    if i % 10 ==0:
        print(i)"""


# In[ ]:


"""params_df2 = pd.Series(params_dict2)
print(len(params_dict2))
params_df2 = params_df2.sort_values(ascending=True)
params_df2[:10]"""


# In[ ]:


model2 = KNeighborsRegressor(n_neighbors = 7,weights ='uniform',metric = 'manhattan',algorithm = 'auto',n_jobs=4)
model2.fit(X_scaled,y2)
#Training error
print("training error for y2 = ", round(rmsle(model2.predict(X_scaled),y2),5))


# In[ ]:


knn_y2_train = model2.predict(X_scaled)
knn_y2_test = model2.predict(test_scaled)
knn_y2_train = pd.Series(knn_y2_train)
knn_y2_test = pd.Series(knn_y2_test)


# # Merging all of them, XGBOOST,EXTRATREES, RIDGE REGRESSION, KNN
# 

# In[ ]:


xgboost_y1_train.name ="xgboost"
xgboost_y1_test.name ="xgboost"
xgboost_y2_train.name ="xgboost"
xgboost_y2_test.name ="xgboost"
ridge_y1_train.name = "ridge"
ridge_y1_test.name = "ridge"
ridge_y2_train.name = "ridge"
ridge_y2_test.name = "ridge"
et_y1_train.name = "et"
et_y1_test.name = "et"
et_y2_train.name = "et"
et_y2_test.name = "et"
knn_y1_train.name = "knn"
knn_y1_test.name = "knn"
knn_y2_train.name = "knn"
knn_y2_test.name = "knn"


# In[ ]:


y1_training = pd.concat([xgboost_y1_train,ridge_y1_train,et_y1_train,knn_y1_train],axis=1)
y1_testing = pd.concat([xgboost_y1_test,ridge_y1_test,et_y1_test,knn_y1_test],axis=1)
y2_training = pd.concat([xgboost_y2_train,ridge_y2_train,et_y2_train,knn_y2_train],axis=1)
y2_testing = pd.concat([xgboost_y2_test,ridge_y2_test,et_y2_test,knn_y2_test],axis=1)


# In[ ]:


print(y1_training.head())


# In[ ]:


average_y1_train = (y1_training['xgboost']+y1_training['ridge']+y1_training['et']+y1_training['knn'])/4
average_y1_test = (y1_testing['xgboost']+y1_testing['ridge']+y1_testing['et']+y1_testing['knn'])/4
average_y2_train = (y2_training['xgboost']+y2_training['ridge']+y2_training['et']+y2_training['knn'])/4
average_y2_test = (y2_testing['xgboost']+y2_testing['ridge']+y2_testing['et']+y2_testing['knn'])/4


# In[ ]:


print("training error of average mode for y1 = ",rmsle(average_y1_train,y1))
print("training error of average mode for y2 = ",rmsle(average_y2_train,y2))


# In[ ]:


average_y1_test.name="bandgap_energy_ev"
average_y2_test.name="formation_energy_ev_natom"


# In[ ]:


#Taking on the main scale
average_y1_test = np.exp(average_y1_test)-1
average_y2_test = np.exp(average_y2_test)-1


# In[ ]:


average_y1_test.head()


# In[ ]:


submission_file = pd.concat([test_id,average_y2_test,average_y1_test],axis=1)
submission_file.to_csv("nomad2018_last.csv",index=False)


# In[ ]:





# In[ ]:




