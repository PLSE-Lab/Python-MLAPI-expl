#!/usr/bin/env python
# coding: utf-8

# Let's try to understand how data is **distributed over time** in [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction) training dataset.

# Dataset is roughly 9GB. So what would we use to load it?
# 
# numpy.loadtxt is known to be awkwardly slow. On the other hand pandas.read_csv is really clumsy in terms of memory consumption.
# I've tried few additional flags, but still can't guarantee to run this code multiple times.

# In[ ]:


import pandas
import numpy as np

dt = { 'acoustic_data': 'i2', 'time_to_failure': 'f8' }
data = pandas.read_csv("../input/train.csv", dtype=dt, engine='c', low_memory=True)

N = data.shape[0]
print("Data size", N)
#print("Data stats", data.describe())


# Meantime, test data chunks all have same size of 150k records and doesn't have any time markers. So unless further clarified by competition organizers, we'll have to make some assumptions about time function for test chunks and training data, and their correllation.
# 
# As stated by organizers in [Data description](https://www.kaggle.com/c/LANL-Earthquake-Prediction/data) and [Additional info](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/77526):
# > The training data is a single, continuous segment of experimental data.
# 
# > The input is a chunk of 0.0375 seconds of seismic data (ordered in time), which is recorded at 4MHz, hence 150'000 data points, and the output is time remaining until the following lab earthquake, in seconds.
# 
# > Both the training and the testing set come from the same experiment.
# 
# Does it mean, that training data also recorded at 4MHz? Probably.
# And that it also consists of 0.0375 second (150'000 data points) chunks? I don't think so. In this kernel, I'll try to show why.
# 
# Let's figure out what does "continuous segment of experimental data" mean in reality.
# 
# I would read it as: **Time interval (and thus difference between "time_to_failure" values) between adjacent data frames is constant**, unless "time_to_failure" values point to 2 different earthquakes.
# 
# Let's take a closer look into time_to_failure sequence, and validate this interpetation.

# In[ ]:


digits = 10

oldValue = data.iloc[0,1]
newValue = data.iloc[1,1]
frame_diff = round(oldValue - newValue, digits)
oldDiff = frame_diff
print("Time to failure", round(oldValue, digits))
for i in range(1, 20000):
    newValue = data.iloc[i,1]
    newDiff = round(oldValue - newValue, digits)
    if oldDiff != newDiff:
        print("Time difference changed from", oldDiff, "to", newDiff, "on frame", i)
        oldDiff = newDiff
    oldValue = newValue
print("Time to failure", round(newValue, digits))


# Code above tests if frame rate is constant or changes over time. As we can see there are gaps before every 4096-th frame.
# 
# And to be clear on durations of these gaps:

# In[ ]:


chunk_diffs = [
    round(data.iloc[0,1] - data.iloc[4096,1], digits),
    round(data.iloc[4095,1] - data.iloc[8191,1], digits)
]
print(chunk_diffs)


# Obviously first continuous data chunk has length of 4095 frames, while next few chunks have length of 4096 frames.
# 
# So, does all dataset consist of chunks of nearly 4096 (2^12) length?
# 
# Lets make few more calculations.

# In[ ]:


M = 4096
C = N//M+1
R = C*M - N
print("Chunk size", M)
print("Number of chunks", C)
print("Number of incomplete chunks", R)


# Looks quite promising.
# 
# So if our theory is true:
# * There are 153600 chunks of sequential data.
# * Chunks have length of **4096 frames**, with 120 missing frames spreaded among all chunks.
# * Time between frames inside chunk is **1.1e-9**.
# * Time between corresponding (first or last) frames of adjucent chunks is either **0.001 or 0.0011**
# 
# Let's validate. Code below **will stop execution if above constrains are not satisfied**. There are, however, few notes to consider:
# * Some "time_to_failure" values in source data are rounded to 1e-9 precision losing last digit of frame duration (which is 1.1e-9)
# * If one of the adjucent chunks has less then 4096 frames, and misses few starting or ending frames, then time beetween first frames might be not exactly equal to either 0.001 or 0.0011.
# 
# Additionally, acoustic data and chunks metadata related to **separate earthquakes saved to individual output files**. This would be valuable outcome for me and anybody willing to perform analysis earthquake-wise and **significanlty reduce size** of source data to load.
# 
# Note: *Chunk metadata being saved consists of first frame index for each chunk, and time difference between first frame and first frame of previous chunk. First chunk stores initial "time_to_failure" instead. This should be enough to restore "time_to_failure" values for every frame.*

# In[ ]:


chunk_digits = 8

max_error = 1e-9
small_error_count = 10
failure_starts = [[0,0]]
failure_index = 0
chunks = np.zeros((C, 2))
chunk_index = 0
startValue = data.iloc[0,1]
chunks[chunk_index] = [0, round(startValue, digits)]
print("Time to", failure_index, "earthquake", round(startValue, digits))
for i in range(1, N):
    j = i - chunks[chunk_index,0]
    newValue = data.iloc[i,1]
    newDiff = round(startValue - newValue, digits)
    frame_error = newDiff - round(frame_diff * j, digits)
    if newDiff < 0:
        print("New earthquake after", chunk_index+1, "chunks", i-failure_starts[failure_index][0], "samples")
        data.iloc[failure_starts[failure_index][0]:i,0].to_csv('failure{0}_data.zip'.format(failure_index), index=False, compression="zip")
        np.savetxt('failure{0}_chunks.csv'.format(failure_index), chunks[failure_starts[failure_index][1]:chunk_index+1], fmt='%d, %f', header='Frame, Time', comments='')
        print("Time to", failure_index, "earthquake", round(startValue, digits))
        failure_index += 1
        chunk_index += 1
        failure_starts.append([i, chunk_index])
        startValue = newValue
        chunks[chunk_index] = [i, round(startValue, digits)]
        print("Time to", failure_index, "earthquake", round(startValue, digits))
    elif round(newDiff, chunk_digits) in chunk_diffs:
        # 
        chunk_index += 1
        chunks[chunk_index] = [i, newDiff]
        startValue = newValue
        if chunk_index % 1000 == 0:
            print("Chunk", chunk_index)
    elif frame_error != 0:
        if small_error_count > 0:
            print("Unexpected frame", failure_index, chunk_index, j, "duration", newDiff, "expected", round(frame_diff * j, digits))
            small_error_count -= 1
        if frame_error > max_error:
            print("Prediction error, stopping execution")
            break

data.iloc[failure_starts[failure_index][0]:,0].to_csv('failure{0}_data.zip'.format(failure_index), index=False, compression="zip")
np.savetxt('failure{0}_chunks.csv'.format(failure_index), chunks[failure_starts[failure_index][1]:], fmt='%d, %f', header='Frame, Time', comments='')
print("Time to", failure_index, "earthquake", round(newValue, digits))


# Missing starting frames for some chunks are completely fine, and shouldn't have noticable impact on predictions quality.
# 
# Time differences between chunks seem to follow nearly cyclic pattern with cycle consisting of 25 chunks. But it sometimes changes throughout training data, so additional investigation might be necessary to make any assumptions about nature of time gaps between chunks.
# 
# None of the gaps is enough to put 0.0375 second (or 150'000 frames) inside. So if training data somehow overlaps with tests data, there will be overlapping series of accoustic data values at least of (nearly) 4096 size.
# 
# Please, note, that if **test data represents long continuous time segments** as stated, while **training data consists only of short continuous chunks**, with seemingly **different frame rate** inside each chunk, it might **significantly affect prediction quality** for solutions trained only on training dataset.

# In[ ]:


failure_count = failure_index + 1
chunk_count = chunk_index + 1
print("Chunk count expected", C, "actual", chunk_count)
for fi in range(failure_count):
    first_frame = failure_starts[fi][0]
    last_frame = int(chunks[(failure_starts[fi+1][1] if fi < failure_count - 1 else chunk_count) - 1, 0])
    first_ttf = data.iloc[first_frame,1]
    last_ttf = data.iloc[last_frame,1]
    rate = (last_frame - first_frame)/(first_ttf - last_ttf)
    print("Earthquake", failure_index, "frame range", first_frame, last_frame, "time range", first_ttf, last_ttf, "measured frame rate", rate, "Hz")


# I hope, that:
# * my research and produced outcome will be useful. 
# * we'll get more insights from organizers shortly about the way training dataset was prepared and it's relation to test data.
