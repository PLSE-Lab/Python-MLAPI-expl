#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import IPython.display as ipd
import audioread
import soundfile as sf
import io
import subprocess
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.gridspec as gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.decomposition import PCA
from joblib import parallel_backend
from ipywidgets import GridspecLayout, Output, HBox, VBox, Label, Layout, Accordion
from sklearn.utils import shuffle
import ipywidgets as widgets


# In[ ]:


# %matplotlib widget


# In[ ]:


TMPDIR = "./tmp"
DATADIR = "../input/birdsong-recognition"
SR=16000
RSEED=2020


# In[ ]:


## 
## converting MP3s to WAV files
##

def getwav(ebird_code, mp3, sr=SR, tmpdir=TMPDIR, datadir=DATADIR):
    wav = '{}/{}/{}/{}.wav'.format(tmpdir, sr, ebird_code, os.path.splitext(mp3)[0])
    if not os.path.exists(wav):
        d = os.path.split(wav)[0]
        if not os.path.exists(d): os.makedirs(os.path.split(wav)[0])
        try:
            cmd = 'ffmpeg -i {}/train_audio/{}/{} -acodec pcm_s16le -ar {} {}'.format(datadir, ebird_code, mp3, sr, wav)
            o = subprocess.check_call(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            raise Exception("ffmpeg failed without code {} and output '{}', '{}'".format(exc.returncode, exc.output, cmd))
    dt = wavfile.read(wav)
    return dt


# In[ ]:


##
## Splitting audio files into 1 sec portions, overlapping by 0.5 sec
##

def sliceaudio(x, sr, f, ebird_code, channel):
    nc, ns = x.shape if len(x.shape)>1 else (1,len(x))
    assert len(channel) == nc, 'invalid channel, len({}) != {} ({})'.format(channel, nc, x.shape)
    xi = range(0, ns-sr//2, sr//2)
    xi = np.array([xi, np.array(xi)+sr]).T
    xi[-1] = [ ns-sr, ns ]
    xn = np.apply_along_axis(lambda a: x[:, a[0]:a[1]], 1, xi)
    dm = np.divmod(xi[:,0].repeat(nc, 0)/SR, 60.)
    xl = pd.DataFrame({'ebird_code':ebird_code,
           'filename':f[:-4],
           'offset':xi[:,0].repeat(nc, 0),
           'channel':np.tile(channel, xi.shape[0]),
           'min':dm[0],
           'sec':dm[1]
          })
    return xn.reshape(-1, sr), xl

##
## Loading all data for the specified bird
##

def loadbirddata(dt, ebird_code, sr=SR):
    xdata = []
    labels = []
    for f in dt[dt['ebird_code']==ebird_code]['filename']:
        sr, x = getwav('aldfly', f)
        assert len(x.shape)==1 or x.shape[1]==2
        ns, nc = x.shape if len(x.shape)>1 else (len(x), 1)
        x = x.reshape(-1, nc).T.astype(float)
        if ns < sr:
            x = np.hstack([x, np.zeros((nc, sr-ns), dtype=float)])
        xn, xl = sliceaudio(x, sr, f, ebird_code, channel=(['left','right'] if nc==2 else ['mono']))
        xdata.append(xn)
        labels.append(xl)
        if nc == 3:
            xm = librosa.core.to_mono(x).reshape(1, ns)
            xn, xl = sliceaudio(xm, sr, f, ebird_code, channel=['mixed'])
            xdata.append(xn)
            labels.extend(xl)
    xdata = np.vstack(xdata).astype(np.single)
    labels = pd.concat(labels)
    return xdata, labels


# In[ ]:


dt_train = pd.read_csv('{}/train.csv'.format(DATADIR))
dt_test = pd.read_csv('{}/test.csv'.format(DATADIR))
dt_sample = pd.read_csv('{}/example_test_audio_summary.csv'.format(DATADIR))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n## Load & shuffle data for Alder Flycatcher\n\nxdata, labels = loadbirddata(dt_train[:1000], 'aldfly')\nxdata, labels = shuffle(xdata, labels, random_state=RSEED)\nxdata.shape, labels.shape")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n## Create MFCCs for all segments\n\nmfcc = np.apply_along_axis(lambda a: librosa.feature.mfcc(y=a, sr=SR).reshape((-1)), 1, xdata)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n## PCA to reduce dimensionality of data; we'll use at least 3 dimensions later\n\npca = PCA(n_components=mfcc.shape[1])\nmfcc2 = pca.fit_transform(mfcc)\nmfcc2 = mfcc2[:, :max(np.argmax(np.cumsum(pca.explained_variance_ratio_) > .8)+1,3)]\nplt.figure(figsize=(7,2))\np = plt.plot(np.cumsum(pca.explained_variance_ratio_))\np[0].axes.set_ylim([0, 1])\np[0].axes.set_title('PCA of MFCC\\nCumulative explained variance ratio'.format(pca.n_components, mfcc.shape[1]))\nplt.show()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n## Use OPTICS to identify outliers\n\nclust = OPTICS(min_samples=.05, cluster_method='xi', xi=.05, min_cluster_size=.05)\nwith parallel_backend('threading'):\n    clust.fit(mfcc2)")


# In[ ]:


## display worst and best samples - worst meaning the distance to a (any) cluster is the largest

def AddAudio(grid, ndxs, dims):
    pos = 0
    for ndx in ndxs:
        label = '{ebird_code} - {filename} - {channel} - {min}:{sec:02.2f} min'.format(**labels.iloc[ndx].to_dict())
        audio = xdata[ndx]
        out = Output()
        with out:
            display(ipd.Audio(audio, rate=SR))
            grid[pos//dims[0], pos%dims[0]] = VBox([Label(value=label), out], layout=Layout(align_items='center'))
        pos = pos + 1

tab = widgets.Tab(children=[GridspecLayout(4,4) for i in range(2)])
tab.set_title(0, 'Best')
tab.set_title(1, 'Worst')
AddAudio(tab.children[0], clust.ordering_[:16], (4,4))
AddAudio(tab.children[1], clust.ordering_[-16:], (4,4))
tab


# In[ ]:


## Plot the cluster (admittedly this isn't terribly useful ...)

X = mfcc2

plt.figure(figsize=(11,8))
ax = plt.subplot(projection='3d')

r = clust.reachability_
r = (r - np.nanmin(r[~np.isinf(r)])) / np.nanmax(r[~np.isinf(r)])
r[np.isinf(r)] = 1
r = np.digitize(r, np.unique(np.percentile(r, np.linspace(0, 100, num=1000)))) 
r = r/r.max()
sp = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=r, marker='.', alpha=0.3, s=5, cmap=plt.cm.get_cmap('RdYlBu'), vmin=0, vmax=1)
plt.colorbar(sp)

plt.show()

