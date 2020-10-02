#!/usr/bin/env python
# coding: utf-8

# ## Full credits to the author [Marcus F](https://www.kaggle.com/friedchips) of [the original notebook](https://www.kaggle.com/friedchips/clean-removal-of-data-drift/).
# 
# #### I have just reorganised and refactored the code for my curiosity and learnings. Added console longs, a few additional graphs and also saved the datasets as `.csv` files. 
# 
# ##### I have also posted a couple of questions/discussion points on the [Comments section](https://www.kaggle.com/friedchips/clean-removal-of-data-drift/comments#819150). Please feel free to answer there as well as comment below.
# 
# Also recommend have a read of this discussion on [What is drift](https://www.kaggle.com/c/liverpool-ion-switching/discussion/133874)?
# 
# ### Find other such refactored notebooks [here](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153653).

# This notebook shows how to remove the drift from the training and test data as cleanly as possible. A clean signal is extremely important, since predictions from any ML models depend strongly on the precise value of each data point. The drift is removed by computing the histograms of small signal batches and matching them to an ideal (non-shifted) histogram. The resulting shifts are already much better than those from e.g. a rolling mean. The shifts are then further smoothed by approximating them with 4th degree polynomials. 
# The resulting clean signal retains the original offset.
# For the pupose of this competition, the "signal groups" (see below) are determined by hand. This could also be done in an automated way (e.g. through analysis of the histograms) in the case of real-world data.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple


# In[ ]:


# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 5.0]


# In[ ]:


def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    return fig, axes
    
def set_axes(axes, use_grid=True, x_val = [0,100,10,5], y_val = [-50,50,10,5]):
    axes.grid(use_grid)
    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)
    axes.set_xlim(x_val[0], x_val[1])
    axes.set_ylim(y_val[0], y_val[1])
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[2] + 1).astype(int)))
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[3] + 1).astype(int)), minor=True)
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[2] + 1).astype(int)))
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[3] + 1).astype(int)), minor=True)


# In[ ]:


def remove_drift_from_data(segm_signal_groups: List,
                           signal: np.ndarray,
                           hist_bins: Tuple[np.ndarray, any],
                           clean_hist: List,
                           s_window: int = 10):
    print('len(segm_signal_groups)', len(segm_signal_groups))
    print('len(signal[n]):', len(signal[0]), f'x {len(segm_signal_groups)- 1}')

    bin_width = np.diff(hist_bins)[0]
    signal_shift = []
    for clean_id in range(len(segm_signal_groups)):

        group_id = segm_signal_groups[clean_id]
        window_shift = []
        prev_s = 0
        window_data = signal[clean_id].reshape(-1, window_size)

        print(
            f'Processing clean_id: {clean_id}, group_id: {group_id}, '
            f'window_data: {window_data[0][:3]}, len(window_data): {len(window_data)}...'
        )

        for w in window_data:
            window_hist = np.histogram(w, bins=hist_bins)[0] / window_size
            window_corr = np.array([np.sum(clean_hist[group_id] * np.roll(window_hist, -s))
                                    for s in range(prev_s - s_window, prev_s + s_window + 1)])
            prev_s = prev_s + np.argmax(window_corr) - s_window
            window_shift.append(-prev_s * bin_width)

        window_shift = np.array(window_shift)
        signal_shift.append(window_shift)

    return signal_shift


# In[ ]:


# using 4th order polynomial feature
def remove_noise_from_data(window_size: int,
                 signal_shift: List[np.ndarray],
                 segm_is_shifted: List[np.ndarray],
                 signal: np.ndarray) -> Tuple[List, List]:
    signal_shift_clean = []
    signal_detrend = []
    for data, use_fit, signal in zip(signal_shift, segm_is_shifted, signal):
        if use_fit:
            data_x = np.arange(len(data), dtype=float) * window_size + window_size / 2
            fit = np.flip(np.polyfit(data_x, data, 4))
            data_x = np.arange(len(data) * window_size, dtype=float)
            data_2 = np.sum([c * data_x ** i for i, c in enumerate(fit)], axis=0)
        else:
            data_2 = np.zeros(len(data) * window_size, dtype=float)

        signal_shift_clean.append(data_2)
        signal_detrend.append(signal + data_2)

    return signal_shift_clean, signal_detrend


# In[ ]:


def create_clean_histogram(signal: np.ndarray, 
                           histogram_chunk_size: int, 
                           model_segments: List) -> List:
    clean_hist = []
    for j, i in enumerate(model_segments):
        clean_hist.append(np.histogram(signal[i], bins=hist_bins)[0])
        clean_hist[-1] = clean_hist[-1] / histogram_chunk_size   # normalize histogram

    return clean_hist


# In[ ]:


def save_dataframe(dataframe: pd.DataFrame,
                   signal_detrend: np.ndarray,
                   openchannel: np.ndarray,
                   filename_with_path: str,
                   force_overwrite=False):
    dataframe['signal'] = np.concatenate(signal_detrend)
    if openchannel:
        dataframe['open_channels'] = np.concatenate(openchannel)
    print("Shape:", dataframe.shape)
    print("Contents:\n", dataframe)
    print()
    print(f'force_overwrite = {force_overwrite}')
    if force_overwrite or (not os.path.exists(filename_with_path)):
        print(f"Saving dataframe to {filename_with_path}.")
        dataframe.to_csv(filename_with_path, index=False, float_format='%.9f',
                                       chunksize=100000, compression='gzip', encoding='utf-8')
    else:
        print(f"{filename_with_path} already exists, not overwriting. Remove it and try again.")


# In[ ]:


window_size = 1000
NOT_APPLICABLE = None
s_window = 10  # maximum absolute change in shift from window to window+1
hist_bins = np.linspace(-4,10,500)
model_segments = [0, 3, 4, 6, 5]


# The data consists of different segments which can be sorted into one of 5 "signal groups":

# In[ ]:


df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
df_test  = pd.read_csv("../input/liverpool-ion-switching/test.csv")


# ## Training and test dataset overview

# In[ ]:


fig, axes = create_axes_grid(1,2,30,5)
# training dataset
set_axes(axes[0], x_val=[0,5000000,500000,100000], y_val=[-5,15,5,1])
axes[0].plot(df_train['signal'], color='darkblue', linewidth=.1);
axes[0].set_title('training')
plt.tick_params(labelsize=14)
# test dataset
set_axes(axes[1], x_val=[0,2000000,100000,10000], y_val=[-5,15,5,1])
axes[1].set_title('test')
axes[1].plot(df_test['signal'], color='darkgreen', linewidth=.1);
plt.tick_params(labelsize=14)


# Visual identification can easily determine a) the signal group and b) whether there is a drift:

# ## Training dataset

# In[ ]:


train_segm_separators = np.concatenate([[0,500000,600000], np.arange(1000000,5000000+1,500000)]) 
## notice [0,500000,600000], to pick up the abbreration in the signals

# [0, 1, 2, 3, 4] = are indices of the signal groups 0 to 4.
train_segm_signal_groups = [0,0,0,1,2,4,3,1,2,3,4] # from visual identification
print("len(train_segm_signal_groups)", len(train_segm_signal_groups))
train_segm_is_shifted = [False, True, False, False, False, False, False, True, True, True, True] # from visual identification
print("len(train_segm_is_shifted)", len(train_segm_is_shifted))
train_signal = np.split(df_train['signal'].values, train_segm_separators[1:-1])
train_opench = np.split(df_train['open_channels'].values, train_segm_separators[1:-1])


# The training data segments 0,3,4,6,5 are the "model segments" for the signal groups 0-5, respectively (clean & no shift):

# In[ ]:


# create clean signal histograms: training dataset
fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[-4,8,1,.1], y_val=[0,0.05,0.01,0.01])

train_clean_hist = []
for j,i in enumerate(model_segments):
    train_clean_hist.append(np.histogram(train_signal[i], bins=hist_bins)[0])
    train_clean_hist[-1] = train_clean_hist[-1] / 500_000 # normalize histogram
    axes.plot(hist_bins[1:], train_clean_hist[-1], label='Data segment (model segment) '+str(i)+', signal group '+str(j));
axes.legend();
axes.set_title("Clean reference histograms for all 5 signal groups");
plt.tick_params(labelsize=16)
plt.legend(prop={'size': 16})


# Let's take a look at the linear shift in segment 1 and compare the histogram of 4 slices of width 1000 (at 0,25000,50000 and 75000) to the clean histogram of segment 0:

# In[ ]:


clean_hist = train_clean_hist


# In[ ]:


segment_starting_points = [0,25000,50000,75000]
axes.plot(hist_bins[1:], clean_hist[0]);
for i in segment_starting_points:
    window_hist = np.histogram(train_signal[1][i:i+window_size], bins=hist_bins)[0] / window_size
    axes.plot(hist_bins[1:], window_hist);


# In[ ]:


def examine_linear_shift_in_train_segment(segment_index: int = 1):
    fig, axes = create_axes_grid(1,1,30,4)
    set_axes(axes, x_val=[-4,2,1,.1], y_val=[0,0.05,0.01,0.01])

    segment_starting_points = [0,25000,50000,75000]
    axes.plot(hist_bins[1:], clean_hist[0]);
    for i in segment_starting_points:
        window_hist = np.histogram(train_signal[segment_index][i:i+window_size], bins=hist_bins)[0] / window_size
        axes.plot(hist_bins[1:], window_hist);


# In[ ]:


#examine_linear_shift_in_train_segment(0) # Data segment (model segment) = 0 (seems fine)
#examine_linear_shift_in_train_segment(2) # Data segment (model segment) = 2 (seems fine)


# In[ ]:


examine_linear_shift_in_train_segment(1) # Data segment (model segment) = 1 (shifted)


# In[ ]:


examine_linear_shift_in_train_segment(3) # Data segment (model segment) = 3 (shifted)


# In[ ]:


examine_linear_shift_in_train_segment(4) # Data segment (model segment) = 4 (shifted)


# In[ ]:


examine_linear_shift_in_train_segment(6) # Data segment (model segment) = 5 (shifted)


# In[ ]:


examine_linear_shift_in_train_segment(5) # Data segment (model segment) = 6 (shifted)


# It's visually clear that the shift can be determined by matching the window histograms to the clean histogram. Now we need to do this automatically for all shifted data segments:

# In[ ]:


clean_hist = create_clean_histogram(train_signal, 
                                    500000, 
                                    model_segments)


# In[ ]:


train_signal_shift = remove_drift_from_data(train_segm_signal_groups,
                                            train_signal,
                                            hist_bins,
                                            clean_hist)


# This results in an already quite clean shift signal:

# In[ ]:


fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[0,5000,500,100], y_val=[-5,1,1,.1])
axes.plot(np.concatenate(train_signal_shift));
axes.set_title("Shift value as determined by histogram matching:");


# Finally, approximation by a 4th order polynomial:

# In[ ]:


train_signal_shift_clean, train_signal_detrend = remove_noise_from_data(
                                    window_size,
                                    train_signal_shift,
                                    train_segm_is_shifted,
                                    train_signal
)


# The final shift and the cleaned signal:

# In[ ]:


fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[0,5000000,500000,100000], y_val=[-6,1,1,.1])
axes.plot(np.concatenate(train_signal_shift_clean));
axes.set_title("Final shift value after polynomial fit");


# In[ ]:


fig, axes = create_axes_grid(1,1,30,5)
set_axes(axes, x_val=[0,5000000,500000,100000], y_val=[-5,15,5,1])
axes.plot(np.concatenate(train_signal_detrend), linewidth=.1);
axes.set_title("Training data without shift");


# Finally, save all info as a csv file:

# In[ ]:


save_dataframe(df_train,
               train_signal_detrend,
               train_opench,
               'train_clean_removed_drift.csv.gz')


# ## Test dataset

# And the same procedure for the test data:

# In[ ]:


df_test  = pd.read_csv("../input/liverpool-ion-switching/test.csv")


# In[ ]:


test_segm_separators = np.concatenate([np.arange(0,1000000+1,100000), [1500000,2000000]])
# [0, 1, 2, 3, 4] = are indices of the signal groups 0 to 4.
test_segm_signal_groups = [0,2,3,0,1,4,3,4,0,2,0,0] # from visual id
print("len(test_segm_signal_groups)", len(test_segm_signal_groups))
test_segm_is_shifted = [True, True, False, False, True, False, True, True, True, False, True, False] # from visual id
print("len(test_segm_is_shifted)", len(test_segm_is_shifted))
test_signal = np.split(df_test['signal'].values, test_segm_separators[1:-1])


# In[ ]:


# Just drawing the graphs for the test dataset out of curiosity, 
# since we don't do the same exercise for it as we did for the training dataset above
# create clean signal histograms: test dataset
fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[-4,8,1,.1], y_val=[0,0.05,0.01,0.01])

test_clean_hist = []
for j,i in enumerate(model_segments):
    test_clean_hist.append(np.histogram(test_signal[i], bins=hist_bins)[0])
    test_clean_hist[-1] = test_clean_hist[-1] / 200_000 # normalize histogram
    axes.plot(hist_bins[1:], test_clean_hist[-1], label='Data segment (model segment) '+str(i)+', signal group '+str(j));
axes.legend();
axes.set_title("Clean reference histograms for all 5 signal groups");
plt.tick_params(labelsize=16)
plt.legend(prop={'size': 16})


# In[ ]:


test_signal_shift = remove_drift_from_data(test_segm_signal_groups,
                                           test_signal,
                                           hist_bins,
                                           clean_hist)  


# In[ ]:


fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[0,2000,100,10], y_val=[-6,1,1,.1])
axes.plot(np.concatenate(test_signal_shift));
axes.set_title("Shift value as determined by histogram matching:");


# In[ ]:


test_remove_shift = [True, True, False, False, True, False, True, True, True, False, True, False]
test_signal_shift_clean, test_signal_detrend = remove_noise_from_data(
                                   window_size,
                                   test_signal_shift,
                                   test_segm_is_shifted,
                                   test_signal
)


# In[ ]:


fig, axes = create_axes_grid(1,1,30,4)
set_axes(axes, x_val=[0,2000000,100000,10000], y_val=[-6,1,1,.1])
axes.plot(np.concatenate(test_signal_shift_clean));
axes.set_title("Final shift value after polynomial fit");


# In[ ]:


fig, axes = create_axes_grid(1,1,30,5)
set_axes(axes, x_val=[0,2000000,200000,10000], y_val=[-5,12,5,1])
axes.plot(np.concatenate(test_signal_detrend), linewidth=.1);
axes.set_title("Test data without shift");


# Finally, save all info as a csv file:

# In[ ]:


save_dataframe(df_test,
           test_signal_detrend,
           NOT_APPLICABLE,
           'test_clean_removed_drift.csv.gz')

