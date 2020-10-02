# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
Created on Thu Nov  2 10:27:40 2017
Updated 01/30/2019 to deal with new location of single subject data following dataset revision
@author: brian roach

This script is intended to import single trial csv data from button pressing 
EEG recordings described in the data overview.  Inspiration (and code) heavily borrowed from 
Alexander Barachant
    (https://www.kaggle.com/alexandrebarachant/common-spatial-pattern-with-mne)
and MNE examples including:
    1. http://martinos.org/mne/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py
    2. http://martinos.org/mne/stable/auto_tutorials/plot_creating_data_structures.html
    
If your goal is to get csv data to a .fif format that can be imported into EEGlab
or fieldtrip within the matlab environment, there are commented out lines to do
this below (However, I haven't re-imported .fif data in the matlab environment,
so use at your own risk)
"""

import numpy as np
import pandas as pd
from mne import EpochsArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs
from mne.viz.topomap import _prepare_topo_plot, plot_topomap
from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from glob import glob

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import welch
from mne import pick_types

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#so that's the template for success, let's start with channel naming...

col_names = pd.read_csv('../input/columnLabels.csv')

ch_names = list(col_names.columns[4:])

# define the channel type, where the first 64 are EEG, next 4 are EOG, then Nose,
#and finally right mastoid (EEG)
ch_type = ['eeg']*64 + ['eog']*4 + ['misc'] + ['eeg']
    
# The sampling rate of the recording
sfreq = 1024  # in Hertz (samples per second)

#print(ch_names)
# read EEG standard montage from mne
montage = read_montage('standard_1005',ch_names)
    
# Initialize required fields
info = create_info(ch_names, sfreq, ch_type, montage)

#for debugging:
#fname = '../predictability/subjects/10.csv'

#define the function:
def creat_mne_epoch_object(fname, info):
    """Create a mne epoch instance from csv file"""
    # Add some more information
    info['description'] = 'dataset from ' + fname
    
    # Trials were cut from -1.5 to 1.5 seconds
    tmin = -1.5

    # Read EEG file
    data = pd.read_csv(fname, header=None)
    
    #and convert it to numpy array:
    npdata = np.array(data)
    
    #the first 4 columns of the data frame are the
    #subject number... subNumber = npdata[:,0]
    #trial number... trialNumber = npdata[:,1]
    #condition number... conditions = npdata[:,2]
    #and sample number (within a trial)... sampleNumber = npdata[:,3]
    
    #sample 1537 is time 0, use that for the event 
    onsets = np.array(np.where(npdata[:,3]==1537))
    conditions = npdata[npdata[:,3]==1537,2]

    #use these to make an events array for mne (middle column are zeros):
    events = np.squeeze(np.dstack((onsets.flatten(), np.zeros(conditions.shape),conditions)))
    
    #now we just need EEGdata in a 3D shape (n_epochs, n_channels, n_samples)
    EEGdata = npdata.reshape(len(conditions),3072,74)
    #remove the first 4 columns (non-eeg, described above):
    EEGdata = EEGdata[:,:,4:]
    EEGdata = np.swapaxes(EEGdata,1,2)
    
    #create labels for the conditions, 1, 2, and 3:
    event_id = dict(button_tone=1, playback_tone=2, button_alone=3)
    
    # create raw object 
    custom_epochs = EpochsArray(EEGdata, info=info, events=events.astype('int'), tmin=tmin, event_id=event_id)
    return custom_epochs

#there are 81 subjects total, only sub 16, 21, 24, 27, 42, 58, 59, 72 available on kaggle
#subjects = (16, 21, 24, 27, 42, 58, 59, 72)
subjects = range(1,24)

auc = []
for i,subject in enumerate(subjects):
    epochs_tot = []
    
    #need to check this against data location on server
    fnames = glob("../input/%d.csv/%d.csv" % (subject, subject)) #glob("../input/%d.csv" % (subject))
    
    #there can be only 1, like highlander:
    fname = fnames[0] 
    session = []
    y = []
  
    # read data 
    custom_epochs = creat_mne_epoch_object(fname, info)
    
    #if this is all the python you can stomach, and you just want to get
    #these data into matlab to use EEGlab or similar toolboxes to do some
    #analysis, save the epochs as .fif and check these links:
    #https://sccn.ucsd.edu/wiki/Talk:M4d
    #http://www.fieldtriptoolbox.org/getting_started/neuromag
    #epochs_fname = os.path.join('./subjects', '%d-epo.fif' % (subject))
    #epochs.save(epochs_fname)
    
    # pick eeg signal
    picks = pick_types(custom_epochs.info,eeg=True)
    
    #could not filter epochs with mne 0.13, upgraded to 0.15 to get this to work:
    # Filter data for alpha frequency and beta band
    # Note that MNE implements a zero phase (filtfilt) filer
    custom_epochs.filter(2,45, picks=picks, method='iir', n_jobs=-1, verbose=False)
    
    #the press then hear a tone trials:
    epochs = custom_epochs['button_tone']
    epochs_tot.append(epochs)
    session.extend([1]*len(epochs))
    y.extend([1]*len(epochs))
    
    #the passive listening trials:
    epochs = custom_epochs['playback_tone']
    
    epochs_tot.append(epochs)
    session.extend([1]*len(epochs))
    y.extend([-1]*len(epochs))
        
    #concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)
    
    # get all the data 
    #X = epochs.get_data()
    
    #get 1s around the tone onset:
    #for debugging: subEpochs = epochs
    X = epochs.crop(tmin=-0.7, tmax=0.299).get_data()
    #but exclude non 'eeg' channels:
    X = X[:,[ch=='eeg' for ch  in ch_type],:]
    
    y = np.array(y)
    
    # run CSP
    csp = CSP(reg='ledoit_wolf')
    csp.fit(X,y)
    
    # compute spatial filtered spectrum
    po = []
    for x in X:
        f,p = welch(np.dot(csp.filters_[0,:].T,x), sfreq, nperseg=X.shape[2])
        po.append(p)
    po = np.array(po)
    
    # prepare topoplot
    _,epos,_,_,_ = _prepare_topo_plot(epochs,'eeg',None)
    
    # plot first pattern
    pattern = csp.patterns_[0,:]
    pattern -= pattern.mean()
    ix = np.argmax(abs(pattern))
    # the parttern is sign invariant.
    # invert it for display purpose
    if pattern[ix]>0:
        sign = 1.0
    else:
        sign = -1.0
    
    fig, ax_topo = plt.subplots(1, 1, figsize=(12, 4))
    title = 'Spatial Pattern'
    fig.suptitle(title, fontsize=14)
    img, _ = plot_topomap(sign*pattern,epos,axes=ax_topo,show=False)
    divider = make_axes_locatable(ax_topo)
    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(img, cax=ax_colorbar)
    
    # plot spectrum
    fix = (f>7) & (f<35)
    ax_spectrum = divider.append_axes('right', size='300%', pad=1.2)
    ax_spectrum.plot(f[fix],np.log(po[y==1][:,fix].mean(axis=0).T),'-r',lw=2)
    ax_spectrum.plot(f[fix],np.log(po[y==-1][:,fix].mean(axis=0).T),'-b',lw=2)
    ax_spectrum.set_xlabel('Frequency (Hz)')
    ax_spectrum.set_ylabel('Power (dB)')
    plt.grid()
    plt.legend(['button tone','playback'])
    plt.title('Subject %d' % subject)
    #plt.show()
    plt.savefig('spatial_pattern_subject_%02d.png' % subject ,bbox_inches='tight')
    
    # run cross validation
    clf = make_pipeline(CSP(),LogisticRegression())
    #cv = LeaveOneLabelOut(session) #Note: I can't get this to work with next line:
    #auc.append(cross_val_score(clf,X,y,cv=cv,scoring='roc_auc').mean())
    #...so do 5 fold cv instead :)
    auc.append(cross_val_score(clf,X,y,cv=5,scoring='roc_auc').mean())
    print("Subject %d : AUC 5-fold cross validation score : %.4f" % (subject,auc[-1]))
