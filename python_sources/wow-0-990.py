#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I HATE PYTHON BECAUSE
# IT HAS NOT BRACKETS 
# %debug

import json
import random;
from sklearn.neighbors import KNeighborsClassifier
# :)
from sklearn.tree import DecisionTreeClassifier
import copy;
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def copyobject(a):
    return copy.deepcopy(a);

def copyjson(a):
    return json.loads(json.dumps(a));


def comparematrixes(a,b):
    out=0;
    for i in range(min(len(a),len(b))):
        for j in range(min(len(a[0]),len(b[0]))):
            if a[i][j]==b[i][j]:
                out+=1
    out/=len(a)*len(a[0]);
    return 1-out;


def plot_task(inp,outp):
        """
        Plots the first train and test pairs of a specified task,
        using same color scheme as the ARC app
        """
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, ax = plt.subplots(1, 2, figsize=(15,15),dpi=30)
        ax[0].imshow(inp, cmap=cmap, norm=norm)
        width = np.shape(inp)[1]
        height = np.shape(inp)[0]
        ax[0].set_xticks(np.arange(0,width))
        ax[0].set_yticks(np.arange(0,height))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].tick_params(length=0)
        ax[0].grid(True)
        ax[0].set_title('Input')
        ax[1].imshow(outp, cmap=cmap, norm=norm)
        width = np.shape(outp)[1]
        height = np.shape(outp)[0]
        ax[1].set_xticks(np.arange(0,width))
        ax[1].set_yticks(np.arange(0,height))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].tick_params(length=0)
        ax[1].grid(True)
        ax[1].set_title('Output')
        
        plt.tight_layout()
        plt.show()
plot_task([[1,2],[1,2]],[[1,1,1],[2,2,2],[3,3,3]])

def removeequalinputdata1(ines,outes):
    newin=[];
    newout=[];
    todelete=[True]*len(ines);
    for i in range(len(outes)-1):
        for j in range(i+1,len(outes)):
            if(ines[i]==ines[j] and outes[i]==outes[j]):
                todelete[j]=False
    lnght=len(ines)
    for i in range(len(outes)):
        if todelete[i]:
            newin.append(ines[i])
            newout.append(outes[i])
    return newin,newout;

def removeequalinputdata2(ines,outes):
    newin=[];
    newout=[];
    todelete=[True]*len(ines);
    for i in range(len(outes)-1):
        for j in range(i+1,len(outes)):
            if(ines[i]==ines[j]):
                todelete[j]=False
    lnght=len(ines)
    for i in range(len(outes)):
        if todelete[i]:
            newin.append(ines[i])
            newout.append(outes[i])
    return newin,newout;


# In[ ]:




    

def proces(parsedjson,filename):
    #random.seed(0);
    ines=[];
    outes=[];
    for i in range(len(parsedjson["train"])):
        vx=copyjson(parsedjson["train"][i]["input"]);
        vi=copyjson(parsedjson["train"][i]["output"]);
        for k1 in range(min(len(vx),len(vi))):
            for k2 in range(min(len(vx[0]),len(vi[0]))):
                dtm=[];
                for k3 in range(-2,2+1,1):
                    for k4 in range(-2,2+1,1):
                        if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            td[vx[k1+k3][k2+k4]]=1
                            dtm+=copyjson(td);
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            td[vi[k1+k3][k2+k4]]=1;
                            dtm+=copyjson(td)
                        else:
                            dtm+=[0,0,0,0,0,0,0,0,0,0,1];
                            dtm+=[0,0,0,0,0,0,0,0,0,0,1];
                ines.append(dtm);
                if(len(vi)>k1 and len(vi[0])>k2 and k1>=0 and k2>=0):
                    outes.append(vi[k1][k2]);
                else:
                    print(len(vi),k1)
                    outes.append(0);
    print(len(ines));
    ines,outes=removeequalinputdata1(ines,outes);
    ines,outes=removeequalinputdata2(ines,outes);
    print(len(ines));
    #print(dates[30]);
    knn = KNeighborsClassifier(n_neighbors = 1);
    ines=json.loads(json.dumps(ines));
    #print(ines);
    knn.fit(ines,outes);
    out=[];
    for i in range(len(parsedjson["test"])):
        thisdone=False;
        vx=copyjson(parsedjson["test"][i]["input"])
        vi=copyjson(parsedjson["test"][i]["input"])
        for U in range(20):
            for k1 in range(len(vx)):
                for k2 in range(len(vx[0])):
                    dtm=[];
                    for k3 in range(-2,2+1,1):
                        for k4 in range(-2,2+1,1):
                            if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                                td = [0,0,0,0,0,0,0,0,0,0,0];
                                td[vx[k1+k3][k2+k4]]=1
                                dtm+=copyjson(td)
                                td = [0,0,0,0,0,0,0,0,0,0,0];
                                td[vi[k1+k3][k2+k4]]=1;
                                dtm+=copyjson(td)
                            else:
                                dtm+=[0,0,0,0,0,0,0,0,0,0,1];
                                dtm+=[0,0,0,0,0,0,0,0,0,0,1];
                    vi[k1][k2]=int(knn.predict([dtm])[0]);
            vx=copyjson(vi)
        b=vx;
        plot_task(parsedjson["train"][0]["input"],parsedjson["train"][0]["output"]);
        plot_task(parsedjson["test"][i]["input"],b);
        out=out+[
            filename.replace('.json','_'+str(i))+','+
          json.dumps(b).replace(', ', '').replace('[[', '|')
            .replace('][', '|').replace(']]', '|')+'\n' ];
    return out;


with open('/kaggle/input/abstraction-and-reasoning-challenge/test/2072aba6.json') as f:
    read_data = json.loads(f.read())
    #print(read_data)
    out = proces(read_data,'070dd51e.json');
    print(out);
    print("done"); 



# Any results you write to the current directory are saved as output.


# In[ ]:


# %debug

import os
if os.path.exists("submission.csv"):
    os.remove("submission.csv")
tofile = [];



for dirname, _, filenames in os.walk('/kaggle/input/abstraction-and-reasoning-challenge/test'):
    for filename in filenames:
        parsed=json.loads(open('/kaggle/input/abstraction-and-reasoning-challenge/test/'+filename, "r").read());
        print(filename);
        tofile=tofile+proces(parsed,filename);
tofile.sort();

fl=open("submission.csv","w+");
fl.write("output_id,output\n")
for ob in tofile:
    fl.write(ob)

