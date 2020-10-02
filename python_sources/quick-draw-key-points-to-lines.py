#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">Code Library, Style, and Links</h1>

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nspan {font-family:'Roboto'; color:black; text-shadow:5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:steelblue;}      \n</style>")


# In[ ]:


import numpy as np,pandas as pd
import os,ast,cv2,h5py,warnings
import tensorflow as tf,pylab as pl
from IPython.display import display,HTML
from IPython.core.magic import register_line_magic
warnings.filterwarnings('ignore')
pl.style.use('seaborn-whitegrid')
style_dict={'background-color':'gainsboro','color':'steelblue', 
            'border-color':'white','font-family':'Roboto'}
fpath='../input/quickdraw-doodle-recognition/train_simplified/'
os.listdir("../input")


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">Data Exploration</h1>

# In[ ]:


I=96 # image size in pixels
S=1 # current number of the label set {1,...,34} -> {1-10,...,331-340}
T=10 # number of labels in one set 
N=10000 # number of images with the same label in the training set
files=sorted(os.listdir(fpath))
labels=[el.replace(" ","_")[:-4] for el in files]
print(labels)


# In[ ]:


def display_drawing(data,n,S):
    for k in range(n):  
        pl.figure(figsize=(10,2))
        pl.suptitle(files[(S-1)*T+k])
        for i in range(5):
            picture=ast.literal_eval(data[labels[(S-1)*T+k]].values[i])
            for x,y in picture:
                pl.subplot(1,5,i+1)
                pl.plot(x,y,'-o',markersize=1,color='slategray')
                pl.xticks([]); pl.yticks([])
            pl.gca().invert_yaxis(); pl.axis('equal');
def get_image(data,lw=7,time_color=True):
    data=ast.literal_eval(data)
    image=np.zeros((280,280),np.uint8)
    for t,s in enumerate(data):
        for i in range(len(s[0])-1):
            color=255-min(t,10)*15 if time_color else 255
            _=cv2.line(image,(s[0][i]+10,s[1][i]+10),
                       (s[0][i+1]+10,s[1][i+1]+10),color,lw) 
    return cv2.resize(image,(I,I))


# In[ ]:


nn=np.random.randint(0,T*N,3)
nn


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">Data Compression</h1>

# In[ ]:


@register_line_magic
def data_compression(s):
    S=int(s)
    data=pd.DataFrame(index=range(N),
                      columns=labels[(S-1)*T:S*T])
    for i in range((S-1)*T,S*T):
        data[labels[i]]=        pd.read_csv(fpath+files[i],
                    index_col='key_id').drawing.values[:N]
    display(data.head(3).T.style.set_properties(**style_dict))
    display_drawing(data,5,S)
    images=[]
    for label in labels[(S-1)*T:S*T]:
        images.extend([get_image(data[label].iloc[i]) 
                       for i in range(N)])
    images=np.array(images,dtype=np.uint8)
    targets=np.array([[]+N*[k] for k in range((S-1)*T,S*T)],
                     dtype=np.int32).reshape(N*T)
    nn=np.random.randint(0,T*N,3)
    ll=labels[targets[nn[0]]]+', '+labels[targets[nn[1]]]+   ', '+labels[targets[nn[2]]]
    pl.figure(figsize=(10,2))
    pl.subplot(1,3,1); pl.imshow(images[nn[0]])
    pl.subplot(1,3,2); pl.imshow(images[nn[1]])
    pl.subplot(1,3,3); pl.imshow(images[nn[2]])
    pl.suptitle('Key Points to Lines: %s'%ll)
    pl.show()
    h5f='QuickDrawImages%d.h5'%S
    with h5py.File(h5f,'w') as f:
        f.create_dataset('images',data=images)
        f.create_dataset('targets',data=targets)
        f.close()
    del data,images,targets


# In[ ]:


get_ipython().run_line_magic('data_compression', '1')


# In[ ]:


get_ipython().run_line_magic('data_compression', '2')


# In[ ]:


get_ipython().run_line_magic('data_compression', '3')


# In[ ]:


get_ipython().run_line_magic('data_compression', '4')


# In[ ]:


get_ipython().run_line_magic('data_compression', '5')

