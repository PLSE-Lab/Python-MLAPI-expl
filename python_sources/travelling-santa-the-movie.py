#!/usr/bin/env python
# coding: utf-8

# <h1>Travelling Santa 2018 Prime Paths - The Movie </h1>

# <h3>Now in kernels this Christmas... </h3>
# 
# This leaked footage follows Santa's path as he travels to all the cities on his list.
# 
# Did Rudolph's elf boffins find the shortest path for the reindeer's "Prime" carrot stopovers? 
# 
# Without them, his reindeer, far from the Pole, overworked, energy depleted, exhausted... may wander off course... as the trip gets longer... they may give up entirely!! 
# 
# Too few carrots. Too much distance.  Too little time.  
# 
# Oh the Cervidae!!  (Not to mention the presents!!)
# 
# Will Santa make it to all the cities in time?  Will the reindeer survive the ordeal? 
# 
# Watch this space!  Same Christmas Time!  Same Reindeer Station! 
# 
# 
# Spoiler Alert - Some reindeer may not make it back to the Pole this year due to lack of carrots. 
# Approach with caution and A BIG BAG of CARROTS, if they are found wandering around your neighbourhood.
# 
#  
# <i>No reindeer were harmed in the making of this film.</i>
# 

# In[ ]:


# First up -
# Add Data source for the kernel output to use for path data e.g. Concorde for 5 hours
# Internet connected for kernel

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sympy   # for Primes

# Input data files are available in the "../input/" directory.
# can list the files in the input directory and subdirectories when add data is used
import os
#print(os.listdir("../input"))
# + Add Data e.g. Concorde for 5 hours kernel output
#print(os.listdir("../input/traveling-santa-2018-prime-paths"))
#print(os.listdir("../input/concorde-for-5-hours")) # V1
#print(os.listdir("../input/not-a-3-and-3-halves-opt"))  # V2
# Any results you write to the current directory are saved as output. 


# In[ ]:


# Data load & prep
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')
#df_sub = pd.read_csv('../input/concorde-for-5-hours/submission.csv') # V2
df_sub = pd.read_csv('../input/not-a-3-and-3-halves-opt/1516094.5810868025.csv') #V2

primeset  = sympy.ntheory.generate.primerange(min(cities.CityId), max(cities.CityId))  

cityprimes = np.zeros((max(cities.CityId)+1), dtype=int )
for p in primeset:
    cityprimes[p] = 1     

cities.loc[:,'isPrime'] = cityprimes 

df_sub = pd.merge(df_sub,cities, how='left', left_on=['Path'], right_on=['CityId']  )
# steps 1-10 every 10th would like prime
steps = (df_sub.index + 1) % 10
steps = np.array(steps, dtype=np.int64)
df_sub.loc[:,'steps'] = steps 
df_sub.loc[:,'hitPrime'] = 0
df_sub.loc[((df_sub['isPrime'] ==1) & (df_sub['steps']==0)  ), 'hitPrime' ] = 1
df_sub.loc[:,'missPrime'] = 0
df_sub.loc[((df_sub['isPrime'] ==1) & (df_sub['steps']!=0)  ), 'missPrime' ] = 1
#df_sub.head()


# In[ ]:


# Set up Segments for frames  

x = df_sub.X.values
y = df_sub.Y.values
hits = df_sub.hitPrime.values
misses = df_sub.missPrime.values

# using 100 for segment length to keep movie size small 
seglength = 100  
segs = []
seghits = []
segmisses = []
carrots = []
missed_carrots = []
totc = []
totmiss = []
maxseg = (len(df_sub.Path)// seglength)
if (len(df_sub.Path) % seglength) != 0:
    maxseg = (len(df_sub.Path)//seglength) + 1
for i in range(0, maxseg ):
    ifr = i*seglength
    ito = ito = ifr + seglength
    if ito > len(df_sub.Path):
        ito = len(df_sub.Path)
    segs.append([x[ifr:ito],y[ifr:ito]])
    seghits.append([x[ifr:ito][hits[ifr:ito]==1], y[ifr:ito][hits[ifr:ito]==1]] )
    carrots.append(len(seghits[i][0]))
    totc.append(sum(carrots[:i+1]))
    segmisses.append([x[ifr:ito][misses[ifr:ito]==1], y[ifr:ito][misses[ifr:ito]==1]] )
    missed_carrots.append(len(segmisses[i][0]))
    totmiss.append(sum(missed_carrots[:i+1]))

n_frames = len(segs)
#print(n_frames)


# In[ ]:


# ffmpeg setup 

# inside a docker get ffmpeg binary to be able to use html5 animation and to save as mp4 & for audio edits, merge video and audio, etc.
# N.B. may have to run twice to unpack - 
#      also date embedded in git build may change 


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./ffmpeg-git-20181225-amd64-static/ffmpeg ]]; then\n  echo 'not found' \n  wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz\n  wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5\n  md5sum -c ffmpeg-git-amd64-static.tar.xz.md5\n  tar xvf ffmpeg-git-amd64-static.tar.xz\nfi")


# In[ ]:



#%%bash
#ls ./ffmpeg-git-20181225-amd64-static


# In[ ]:


# imports and setup for matplotlib animation

# set ffmpeg path before animation import - N.B. include ffmpeg not just dir
from matplotlib import pyplot as plt
plt.rcParams["animation.ffmpeg_path"] = '/kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg'
plt.rcParams["animation.html"] = "html5" 
plt.rcParams["animation.embed_limit"] = 200.0   # default is 20.0


from IPython.display import HTML
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=10, metadata=dict(artist='Rudolph'), bitrate=-1 ) # fps=10, ,extra_args=['-vcodec', 'libx264']) 
# defaults will use interval for fps, if -1  matplotlib chooses best bitrate - works best here

import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.path as mpath
from matplotlib import markers
from matplotlib import colors


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# custom markers - cut wedg for carrots (prime on 0 step)

halfcirc = mpath.Path.unit_circle_righthalf()
wedg = mpath.Path.wedge(60,120)
wverts = np.concatenate([halfcirc.vertices, wedg.vertices[::-1, ...]])
wcodes = np.concatenate([halfcirc.codes, wedg.codes])
cut_wedg = mpath.Path(wverts, wcodes)


# In[ ]:


# animation setup

def f(t):
    x=segs[t][0]  
    y=segs[t][1]  
    return x,y

def f2(t):
    x2=seghits[t][0]
    y2=seghits[t][1]
    return x2,y2
    
def f3(t):
    x3=segmisses[t][0]
    y3=segmisses[t][1]
    return x3,y3

fig, ax = plt.subplots(figsize=(12.8,12.8))   # 20,20
fig.suptitle('          Travelling Santa 2018 Prime Paths          ', color = 'r',fontsize=14, linespacing=14)
plt.close()   # close plot to not display an empty plot 
ax.set_aspect('equal')
ax.set_xlim(0,5100)
ax.set_ylim(0,3400)
line = Line2D([],[])
ax.add_line(line)

def init_f():    
    line = Line2D([],[])
    line1 = Line2D([],[])
    line2 = Line2D([],[])
    ax.add_line(line)
    ax.add_line(line1)
    ax.add_line(line2)


def update(t):
    x,y = f(t)
    # optionally clear axes and reset limits
    #plt.gca().cla() 
    #ax.set_aspect('equal')
    #ax.set_xlim(0,5100)
    #ax.set_ylim(0,3400)
    line = Line2D(x,y,lw=2, color='tab:cyan',  marker=7, markevery=10, markerfacecolor= 'tab:gray' )
    ax.add_line(line)
        
    x3,y3 = f3(t) 
    if len(x3)!=0:
        line3 = Line2D(x3,y3, ls='None', marker=7,  markevery=1, markerfacecolor='k'  ) 
        ax.add_line(line3)  
    
    x2,y2 = f2(t)
    if len(x2)!=0:
        line2 = Line2D(x2,y2,ls='None', markevery=1, marker=cut_wedg, markersize=8, markeredgecolor='g', markerfacecolor='tab:orange')
        ax.add_line(line2)
        
    # review spacing in title     
    ax.set_title(' Frame: ' + str(t) + '      Carrots (orange):  ' + str(carrots[t]) + '  Total Carrots:  ' + str(totc[t])  
                 + '   |    Missed 0 Primes (black):  ' + str(missed_carrots[t]) + '  Total Missed:  ' 
                 + str(totmiss[t]), color='g', fontweight="bold", fontsize=14 )

ani = animation.FuncAnimation(fig, update, init_func=init_f, frames=n_frames, interval=100, blit=False, repeat=False) # fargs=None, repeat_delay=10 ,repeat=False)
# interval in ms so depends how fast to set e.g. 100,200,500? set slow enough to see animation


# In[ ]:


# to show animation in notebook as html5 video - 
#  (note this starts running as soon as notebook is opened) 

#HTML(ani.to_html5_video()) #  N.B. this needs ffmpeg to work

# clear & reset plot after for animation save after to get fresh start
#plt.gca().cla()  # if to_html5_video run clear before save 
#plt.close()   # close plot to not display an empty plot
#fig, ax = plt.subplots(figsize=(12.8,12.8))   # 20,20
#fig.suptitle('Travelling Santa 2018 Prime Paths', color = 'r',fontsize=14, linespacing=14)
#plt.close()   # close plot to not display an empty plot 
#ax.set_aspect('equal')
#ax.set_xlim(0,5100)
#ax.set_ylim(0,3400)
#line = Line2D([],[])
#ax.add_line(line)


# In[ ]:


# save animation file

ani.save("travelling_santa.mp4", writer=writer)  


# In[ ]:


# to check file saved 
#print(os.listdir("/kaggle/working/"))


# In[ ]:


# Audio for movie creation

# atm these are the downloads - codes may change, works to open this link in another window, may want to rename 
# TODO use bs to get from span class="playicn"
# http://freemusicarchive.org/music/Various_Artists_from_Dawn_of_Sound/Voices_of_Christmas_Past
# get the codes for each to use on wget, then they will be the same on commit

# atm these are the downloads - codes may change, works to open this link in another window 
# http://freemusicarchive.org/music/Various_Artists_from_Dawn_of_Sound/Voices_of_Christmas_Past
# get the codes for each to use on wget, then they will be the same on commit

# On a good old time sleigh ride [1913] by Peerless Quartet  
#### https://freemusicarchive.org/music/download/53ff20190913a4f6fefa0fd248dcabfac06a5130
#### https://freemusicarchive.org/music/download/63a63f4e6ee7ec1c71916ebe41d9e250aea2ccd4

#https://freemusicarchive.org/music/download/826050a8b9a82b57b8bffd6bd96a284b7e0aad74

# Sleigh ride party / jingle bells [1898] by Edison Male Quartet   
#### https://freemusicarchive.org/music/download/1b94713d23b65f2588019fb530e10a11307d8b1a
#### https://freemusicarchive.org/music/download/5e1b73ab53ad38e75a27c86bd0d6f72b1a1d35ae

#https://freemusicarchive.org/music/download/5e9d25ecc1fffb4e6388649807664263f70b672e

# Chinese Dance, Dance of the Mirilitons (from the Nutcracker) [1913] by Victor Herbert Orchestra 
#### https://freemusicarchive.org/music/download/26fc3aa9132ffe2e90df4c42ec10faa30f376e28
#### https://freemusicarchive.org/music/download/84ce2911fc4e55de4b7a8ac5f622c8092a79060b

#https://freemusicarchive.org/music/download/922dd6d36f4da17e2c9ec8133a85aaf0d27e93d1


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'wget https://freemusicarchive.org/music/download/826050a8b9a82b57b8bffd6bd96a284b7e0aad74')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'wget https://freemusicarchive.org/music/download/5e9d25ecc1fffb4e6388649807664263f70b672e')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'wget https://freemusicarchive.org/music/download/922dd6d36f4da17e2c9ec8133a85aaf0d27e93d1')


# In[ ]:


# to check downloads
#print(os.listdir("/kaggle/working/"))


# In[ ]:


#%%bash
#rmdir ./ffmpeg-git-20181225-amd64-static


# In[ ]:


# imports for audio,video 

from IPython.display import display,Audio
import io
import base64


# In[ ]:


# to  test audio -   this works provided download is OK
#audio = io.open('/kaggle/working/84ce2911fc4e55de4b7a8ac5f622c8092a79060b', 'r+b').read()
#encoded = base64.b64encode(audio)
#HTML(data='''<audio alt="test" controls>
#                <source src="data:audio/mp3;base64,{0}" type="audio/mp3" />
#             </audio>'''.format(encoded.decode('ascii')))


# In[ ]:


# some audio editing with ffmpeg 
# get a clip edit for a section of audio then concat with another audio at start/end, e.g., use to extend audio to match video length


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i /kaggle/working/5e9d25ecc1fffb4e6388649807664263f70b672e -af atrim=140:150  sleigh_edit.mp3')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i /kaggle/working/922dd6d36f4da17e2c9ec8133a85aaf0d27e93d1 -af atrim=0:14  nut_edit.mp3')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i  "concat:/kaggle/working/sleigh_edit.mp3|/kaggle/working/826050a8b9a82b57b8bffd6bd96a284b7e0aad74" -codec copy out_sleigh_mix.mp3')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i  "concat:/kaggle/working/922dd6d36f4da17e2c9ec8133a85aaf0d27e93d1|/kaggle/working/nut_edit.mp3" -codec copy out_nut_mix.mp3')


# In[ ]:


# to test audio mix 
#audio = io.open('/kaggle/working/out_sleigh_mix.mp3', 'r+b').read()
#encoded = base64.b64encode(audio)
#HTML(data='''<audio alt="test" controls>
#                <source src="data:audio/mp3;base64,{0}" type="audio/mp3" />
#             </audio>'''.format(encoded.decode('ascii')))


# In[ ]:


# now to make the movies with animation saved and audio edits


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i /kaggle/working/travelling_santa.mp4 -i /kaggle/working/out_sleigh_mix.mp3 -codec copy sleigh_mix_travelling_santa2.mp4')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '  /kaggle/working/ffmpeg-git-20181225-amd64-static/ffmpeg -i /kaggle/working/travelling_santa.mp4 -i /kaggle/working/out_nut_mix.mp3 -codec copy nutcracker_mix_travelling_santa2.mp4')


# Start  the movie!!  (also available for download from here via right click  or from output for kernel as usual)   
# 
# <h2>Travelling Santa - The Movie </h2>

# In[ ]:


# this works note controls like for HTML5 are hidden  
video = io.open('/kaggle/working/sleigh_mix_travelling_santa2.mp4', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="travelling santa" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))


# <h2>Travelling Santa - The Movie (Nutcracker's Cut) </h2>

# In[ ]:


# this works note controls like for HTML5 are hidden  
video = io.open('/kaggle/working/nutcracker_mix_travelling_santa2.mp4', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="travelling santa (nutcracker)" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))


# <h3>Credits </h3> 
# 
# <b>Path data</b> courtesy of Concorde 5hr kernel  https://www.kaggle.com/blacksix/concorde-for-5-hours    V1
# <br>
# <b>Path data</b> courtesy of Not a 3 and 3 Halves kernel https://www.kaggle.com/kostyaatarik/not-a-3-and-3-halves-opt   V2 <i>(even more carrots!!)</i>
# 
# <b>ffmpeg</b> with thanks to http://johnvansickle.com/ffmpeg/
# 
# <b>Audio tracks - Attribution-Noncommercial-Share Alike 2.5 Canada</b>
# http://freemusicarchive.org/music/Various_Artists_from_Dawn_of_Sound/Voices_of_Christmas_Past
# 
# On A Good Old Time Sleigh Ride [1913] by Peerless Quartet
# 
# Sleigh Ride Party, Jingle Bells [1898] by Edison Male Quartette
# 
# Chinese Dance, Dance of the Mirilitons (from the Nutcracker) [1913] by Victor Herbert Orchestra
# 
# 
