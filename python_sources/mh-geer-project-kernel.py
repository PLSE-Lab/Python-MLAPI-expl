#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import inf
import array
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from random import randint
import pandas as pd  
from matplotlib import pyplot as plt
ids=50
N=50
print('intializing nodes for the network...')
ids=N
df_node = pd.DataFrame(columns = ['x' , 'y', 'z' , 'energy','isClusterHead' , 'amplifier_dBm' , 'isDown'])
while(ids!=1):
        
    df_node=df_node.append({'x' : randint(0,4*N) , 'y' : randint(0,4*N) , 'z' : randint(0,4*N) ,'energy': 0.1 ,'isClusterHead': False,'amplifier_dBm':-100 ,'isDown':False} , ignore_index=True)
    #print(ids)
    #plt.scatter(df_node.x,df_node.y,color='k')
    ids-=1
df_node=df_node.append({'x' : randint(0,N*4) , 'y' : randint(0,N*4), 'z' : randint(0,4*N),'energy': 0.1,'isClusterHead': False,'amplifier_dBm':-100,'isDown':False} , ignore_index=True)
plt.scatter(df_node.x,df_node.y,color='blue')
plt.scatter(df_node.iloc[-1]['x'],df_node.iloc[-1]['y'],color='red')
plt.show()
print('Done.');


# In[ ]:


#clustering part
from sklearn.cluster import KMeans
df_nodeminus =df_node.drop(df_node.index[-1])
df_nodexy = df_nodeminus.drop(['z' , 'energy','isClusterHead' , 'amplifier_dBm' , 'isDown'], axis=1)

X = np.array(df_nodexy.astype(float))
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
prediction = kmeans.predict(X)

df_prediction = pd.DataFrame(data=prediction, columns=['clusters'])
df_prediction.loc[N-1] = 'nan'
   
df_prediction
   


# In[ ]:


df_node.loc[:, 'clusters'] = df_prediction
df_node.head()


# In[ ]:


df_node.head()


# In[ ]:


Target=df_node.groupby(["clusters"])
Target.describe().head()
df_node_clusters=list(Target)[3][1]
df_node_clusters


# In[ ]:


import random
# Create plot
fig = plt.figure()
for i in range(0,5):
    x=list(Target)[i][1]['x']
    y=list(Target)[i][1]['y']
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    group='cluster'+ str(i)
    if(i==4):
        group='Base_station'
        color='red'
    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('clustering_plot')

plt.show()


# In[ ]:


#cluster heads declaration
def electionCH(nodes):#done electing max energie
    df_nodem =df_node.drop(df_node.index[-1])
    df_node_max_index_energy=df_nodem.groupby("clusters").energy.idxmax().rename("maxindexenergy").reset_index()
    cl_image = np.array(df_node_max_index_energy["maxindexenergy"].astype(int))
    df_node['isClusterHead'] = False
    cl_image=list(cl_image)
    df_node.loc[cl_image,'isClusterHead'] = ['H'+'1','H'+'2', 'H'+'3', 'H'+'4']
    return cl_image


# In[ ]:


cl_image=electionCH(df_node)
cl_image


# In[ ]:



df_node


# In[ ]:


cl_image.append(N-1)
cl_image


# In[ ]:


cl_image


# In[ ]:


import math as math
def dist(index1,index2):
    return math.sqrt((df_node.loc[index1,'x']-df_node.loc[index2,'x'])**2+(df_node.loc[index1,'y']-df_node.loc[index2,'y'])**2+(df_node.loc[index1,'z']-df_node.loc[index2,'z'])**2) 
 


# In[ ]:


df_dist = pd.DataFrame(columns = ['H1' , 'H2', 'H3' , 'H4'])
df_dist=df_dist.append({'H1' : dist(cl_image[0],49) , 'H2' : dist(cl_image[1],49) , 'H3' : dist(cl_image[2],49) ,'H4':dist(cl_image[3],49)} , ignore_index=True)


# In[ ]:


df_dist


# In[ ]:


#given values for the problems
n_ants = 1
n_nodes = len(cl_image)
m = n_ants
n = n_nodes
alpha = 1     #pheromone factor
beta = 1       #visibility factor
E=list(df_node.loc[cl_image,'energy'])      #initial energy of a sonsor node
Eelec=5*(10**(-8))       #  [j/bit]
efs=10**(-11)   # [j/bit/m^2]
emp=1.3*(10**(-15))  # [j/bit/m^4]
d0=78.7        #m
packetsize=2000   # bit
Ebs0=1000    # j
taux0=1     #initial pheromone value
print(E)


# In[ ]:


import math as math
e=np.ones((n,n))
global Etx
Etx=np.zeros((n,n))
d0=math.sqrt(efs/emp)
global Erx
Erx=Eelec*packetsize
for i in range(n):
    for j in range(n):
        if dist(cl_image[i],cl_image[j])<=d0:
            Etx[i,j]=packetsize*efs*(dist(cl_image[i],cl_image[j])**2)+Eelec*packetsize
        else:
            Etx[i,j]=packetsize*emp*(dist(cl_image[i],cl_image[j])**4)+Eelec*packetsize
        
        e[i,j]=Etx[i,j]+Erx
print(e)        
            
    
    


# In[ ]:


def emitBits(i,j,df_node,cl_image):
        if (df_node.loc[cl_image[i],'energy']-Etx[i,j]>=0 ):
                df_node.loc[cl_image[i],'energy']-=Etx[i,j]
        else:
            df_node.loc[cl_image[i],'isDown']=True
def recBits(j,df_node,cl_image):
    if (df_node.loc[cl_image[j],'energy']-Erx>=0):
        df_node.loc[cl_image[j],'energy']-=Erx
    else:
        df_node.loc[cl_image[j],'isDown']=True


# In[ ]:


#calculating the visibility 

visibility = E/e
visibility[visibility == inf ] = 0
print(E)
print(e)
print(visibility)


# In[ ]:


#intializing pheromne present at the paths to the nodes

pheromne = taux0*np.ones((n,n))

#intializing the rute of the ants with size rute(n_ants,n_nodes) 

print(pheromne)


# In[ ]:


df_routing_table = pd.DataFrame(columns = ['Neihborj' , 'e(3,j) en mj', 'E(j) en mj' , 'visibility(3,j)','pheromone(3,j)'])
df_routing_table=df_routing_table.append({'Neihborj' : 'H1' , 'e(3,j) en mj' : e[2,0]*(10**(3)) , 'E(j) en mj' : E[0]*(10**(3)) ,'visibility(3,j)':visibility[2,0],'pheromone(3,j)': pheromne[2,0]} , ignore_index=True)
df_routing_table=df_routing_table.append({'Neihborj' : 'H2', 'e(3,j) en mj' : e[2,1]*(10**(3)) , 'E(j) en mj' : E[1]*(10**(3)) ,'visibility(3,j)':visibility[2,1],'pheromone(3,j)': pheromne[2,1]} , ignore_index=True)
df_routing_table=df_routing_table.append({'Neihborj' : 'H4' , 'e(3,j) en mj' : e[2,3]*(10**(3)) , 'E(j) en mj' : E[3]*(10**(3)) ,'visibility(3,j)':visibility[2,3],'pheromone(3,j)': pheromne[2,3]} , ignore_index=True)
df_routing_table=df_routing_table.append({'Neihborj' : 'H5' , 'e(3,j) en mj' : e[2,4]*(10**(3)) , 'E(j) en mj' : E[4]*(10**(3)) ,'visibility(3,j)':visibility[2,4],'pheromone(3,j)': pheromne[2,4]} , ignore_index=True)
df_routing_table=df_routing_table.append({'Neihborj' : 'BS' , 'e(3,j) en mj' : e[2,5]*(10**(3)) , 'E(j) en mj' : E[5]*(10**(3)) ,'visibility(3,j)':visibility[2,5],'pheromone(3,j)': pheromne[2,5]} , ignore_index=True)


# In[ ]:


df_routing_table


# In[ ]:


cl_image


# In[ ]:


n


# In[ ]:


bs=n
CHs=[3,2,1,4]
for ite in range(3):#8 rounds
    #rute = []
    
    for ch in (CHs):
        tab=[]
        rute = []
        rute.append(ch)        #initial starting  positon of every ants '1' i.e node '1'
        visibility = E/e
        for i in range(m):

            temp_visibility = np.array(visibility)         #creating a copy of visibility
            j=0
            while True :


                combine_feature = np.zeros(n)     #intializing combine_feature array to zero
                cum_prob = np.zeros(n)            #intializing cummulative probability array to zeros

                cur_loc = int(rute[j]-1)        #current node of the ant
                
                temp_visibility[:,cur_loc] = 0     #making visibility of the current node as zero

                p_feature = np.power(1/pheromne[cur_loc,:],beta)         #calculating pheromne feature 
                v_feature = np.power((temp_visibility[cur_loc,:]),alpha)  #calculating visibility feature

                p_feature = p_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
                v_feature = v_feature[:,np.newaxis]                     #adding axis to make a size[5,1]

                combine_feature = np.multiply(p_feature,v_feature)     #calculating the combine feature
                            
                total = np.sum(combine_feature)                        #sum of all the feature

                probs = combine_feature/total   #finding probability of element probs(i) = comine_feature(i)/total
                
                node=probs.argmax()+1     #finding the next node having probability higher then random(r) 
                
                emitBits(cur_loc,node-1,df_node,cl_image)
                recBits(node-1,df_node,cl_image)
                
                tab.append(e[int(cur_loc),int(node)-1])
                rute.append(node)             #adding node to route 
                j+=1
                if bs == node:
                     break


        best_route =rute
        #pheromone update
       # for i in range(0,len(best_route)-1):
           # realr.append(cl_image[int(best_route[i]-1)]+1)
           # tab.append(e[int(best_route[i])-1,int(best_route[i+1])-1])
        #realr.append(cl_image[int(best_route[-1]-1)]+1)  
        Emin_loc = np.argmin(tab)             #finding location of minimum of energy_cost
        Emin = tab[Emin_loc] 
        Eavr=sum(tab)/((sum([x * y for x, y in zip(rute, rute)])**(1/2))-1)
        mm=0.1
        EB=Eavr-Emin+mm

        j=bs
        x=len(best_route)-2
        i=int(best_route[x])
        while True :
            dt = EB/E[i-1]
            pheromne[i-1,j-1] = pheromne[i-1,j-1] + dt 
            j=i
            x-=1
            i=int(best_route[x])
            if(j==best_route[0]):
                break

        print('route at the end :'+str(ite))    
        print(best_route)
        E=list(df_node.loc[cl_image,'energy'])     
    #cl_image=electionCH(df_node)
    #cl_image.append(49)


# In[ ]:


def LB(E):
    return 1/(max(E)-min(E))


# In[ ]:


LE=[(0.1-3*Etx[0,4]),(0.1-3*Etx[1,4]),(0.1-3*Etx[2,4]),(0.1-3*Etx[3,4]),(0.1-12*Erx)]
LE


# In[ ]:


0.1-3*Etx[2,3]


# In[ ]:



df_LB_table = pd.DataFrame(columns = [ 'protocol' , 'LB en (j^-1)'])
df_LB_table=df_LB_table.append({'protocol' : 'LEACH','LB en (j^-1)':LB(LE) } , ignore_index=True)
df_LB_table=df_LB_table.append({'protocol' : 'MH-GEER' , 'LB en (j^-1)' : LB(E)} , ignore_index=True)

df_LB_table


# In[ ]:


print(LB(E))


# In[ ]:


df_node


# In[ ]:


e5=E[5-1]*(10**(3))
e2=E[2-1]*(10**(3))
e6=E[6-1]*(10**(3))
print(e5)
print(e2)
print(e6)


# In[ ]:


df_path_table = pd.DataFrame(columns = ['visited nodes' , 'e(i,j) en mj', 'E(j) en mj' ])
df_path_table=df_path_table.append({'visited nodes' : 'H'+str(best_route[0]) , 'e(i,j) en mj' : '-' , 'E(j) en mj' : E[best_route[0]-1]*(10**(3))} , ignore_index=True)
for i in range(1,len(best_route)):
        df_path_table=df_path_table.append({'visited nodes' : 'H'+str(best_route[i]) , 'e(i,j) en mj' : e[best_route[i-1]-1,best_route[i]-1]*(10**(3)) , 'E(j) en mj' : E[best_route[i]-1]*(10**(3))} , ignore_index=True)
df_path_table      


# In[ ]:



df_energy_table = pd.DataFrame(columns = [ 'protocol' , 'E(H1) en mj', 'E(H2) en mj' ,'E(H3) en mj', 'E(H4) en mj'])
df_energy_table=df_energy_table.append({'protocol' : 'LEACH','E(H1) en mj':( LE[0])*(10**(3)),'E(H2) en mj': ( LE[1])*(10**(3)),'E(H3) en mj': ( LE[2])*(10**(3)) ,'E(H4) en mj': ( LE[3])*(10**(3))} , ignore_index=True)
df_energy_table=df_energy_table.append({'protocol' : 'MH-GEER' , 'E(H1) en mj' : E[0]*(10**(3)), 'E(H2) en mj' : E[1]*(10**(3)), 'E(H3) en mj' : E[2]*(10**(3)), 'E(H4) en mj' : E[3]*(10**(3))} , ignore_index=True)

df_energy_table


# In[ ]:


plt.style.use('ggplot')
n = 2
y1 = (85.973540,94.506605)
y2 = (84.336549,94.915188)
y3=(84.386358,90.205508)

fig, ax = plt.subplots()
#index = np.arange(n)
index = np.array([0,0.2])
bar_width = 0.03
opacity = 0.9
ax.bar(index, y1, bar_width, alpha=opacity, color='b',
                label='H5')
ax.bar(index+bar_width, y2, bar_width, alpha=opacity, color='g',
                label='H2')
ax.bar(index+2*bar_width, y3, bar_width, alpha=opacity,
       color='r', label='H6')
ax.set_ylabel('Residual Energy(mj)')
ax.set_title('Residual Energy of nodes H5,H2 and H6')
ax.set_xticks(index+ bar_width )
ax.set_xticklabels(('LEACH','MH-GEER'
    ))
ax.set_ylim([math.ceil(80-0.5*(20)), math.ceil(90+0.5*(20))])
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
#ax.legend(ncol=3)
plt.show()


# In[ ]:


index


# In[ ]:


#makes the data84.,
y1 = [85.973540,94.506605]
y2 = [336549,94.915188]
y3=[84.386358,90.205508]

data = [y1,y2,y3]
X = np.arange(2)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)


# In[ ]:


x=[5,4]
y=['LEACH','MH-GEER']
plt.hist([y,x], color='green')
plt.xlabel('LEACH')
plt.ylabel('Residual Energy(mj)')
plt.title('Residual Energy of nodes H1,H2 and H3')


# In[ ]:



plt.figure(figsize=[10,8])
l = 300
m = 3000
plt.hist(df['protein'])
plt.title("Protein")
plt.show()
k= 3000
b, bins, patches = plt.hist([l, m])


# In[ ]:


print(pheromne)


# In[ ]:


pheromne

