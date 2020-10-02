#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly
import plotly.graph_objects as go


# In[ ]:


#RMS per 1 second - RMS quantifies the destructive value of vibration

import math  
#Function that Calculate Root Mean Square  
def rmsValue(arr, n): 
    square = 0.0
    mean = 0.0
    root = 0.0
      
    #Calculate square 
    for i in range(0,n): 
        square += (float(arr[i])**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = math.sqrt(mean) 
      
    return root 
#print(df["D2"])
import numpy as np
from scipy.fftpack import fft



for dirname, _, filenames in os.walk('C:/Users/Daniel/Documents/Projects/DOST/CRADLE/IMS/1st_test/'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        file1 = open(os.path.join(dirname, filename), 'r') 
        Lines = file1.readlines() 
        #print(len(Lines))
        listA = []
        listB = []
        t = 0
        for line in Lines: 
            #print(f"{line.strip()}") 
            listA.insert(t,float(line.strip().split()[0]))
            listB.insert(t,(t,float(line.strip().split()[0])))
            t = t+1
            #print(line.strip().split()[0])
        #print(f"{filename} length: {len(listA)}")
        N = len(listA)
        frequency = np.linspace (0.0, N, N)
        freq_data = fft(listA)
        y = 2.0/N * np.abs(freq_data[0:N])
        plt.plot(frequency, y)
        plt.title('Frequency domain Signal')
        plt.xlabel('Frequency in Hz')
        plt.ylabel('Amplitude')
        plt.show()
        df = pd.DataFrame(listB, columns = ['index', 'data']) 
        plt.plot(df['index'], df['data'])
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
        print(f"FFT SUM:{np.sum(y)} MEAN: {np.mean(y)} STD: {np.std(y)} RMS: {rmsValue(listA, len(listA))}")
        plotly.offline.plot(df,include_plotlyjs=False, output_type='div')
        #print(listA)
        #df = pd.DataFrame(listB, columns = ['index', 'data']) 
        #df_long=pd.melt(df, id_vars=['index'], value_vars=['data']) # 'price', 'nasdaq'])
        #fig = px.line(df_long, x='index', y='value', title='Time Series with Rangeslider', color='variable')
        #fig.update_xaxes(rangeslider_visible=True)
        #fig.show()

        #print(listA)


# In[ ]:


file1 = open('C:/Users/Daniel/Documents/Projects/DOST/CRADLE/IMS/1st_test/2003.10.22.12.09.13', 'r') 
Lines = file1.readlines() 
listA = []

t = 0
for line in Lines: 
    #val = line.strip().split()[0]
    listA.insert(t, (t, line.strip().split()[0], line.strip().split()[1], line.strip().split()[2], line.strip().split()[3]))
    #print(val) 
    t = t +1
#print(listA)
df = pd.DataFrame(listA, columns = ['index', 'D1', 'D2', 'D3', 'D4']) 
  
print(df)  

df_long=pd.melt(df, id_vars=['index'], value_vars=['D2']) # 'price', 'nasdaq'])

fig = px.line(df_long, x='index', y='value', title='Time Series with Rangeslider', color='variable')

fig.update_xaxes(rangeslider_visible=True)



fig.show()

import math  
#Function that Calculate Root Mean Square  
def rmsValue(arr, n): 
    square = 0.0
    mean = 0.0
    root = 0.0
      
    #Calculate square 
    for i in range(0,n): 
        square += (float(arr[i])**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = math.sqrt(mean) 
      
    return root 
#print(df["D2"])
print(rmsValue(df["D2"], len(df["D2"])))  


# In[ ]:


file1 = open('C:/Users/Daniel/Documents/Projects/DOST/CRADLE/IMS/1st_test/2003.11.21.22.46.56', 'r') 
Lines = file1.readlines() 
listA = []

t = 0
for line in Lines: 
    #val = line.strip().split()[0]
    listA.insert(t, (t, line.strip().split()[0], line.strip().split()[1], line.strip().split()[2], line.strip().split()[3]))
    #print(val) 
    t = t +1
#print(listA)
df = pd.DataFrame(listA, columns = ['index', 'D1', 'D2', 'D3', 'D4']) 
  
print(df)  



df_long=pd.melt(df, id_vars=['index'], value_vars=['D2']) # 'price', 'nasdaq'])

fig = px.line(df_long, x='index', y='value', title='Time Series with Rangeslider', color='variable')

fig.update_xaxes(rangeslider_visible=True)



fig.show()


import math  
#Function that Calculate Root Mean Square  
def rmsValue(arr, n): 
    square = 0.0
    mean = 0.0
    root = 0.0
      
    #Calculate square 
    for i in range(0,n): 
        square += (float(arr[i])**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = math.sqrt(mean) 
      
    return root 
#print(df["D2"])
print(rmsValue(df["D2"], len(df["D2"])))  


# In[ ]:


pip install scipy


# In[ ]:


# Make plots appear inline, set custom plotting style
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#plt.style.use('style/elegant.mplstyle')
import numpy as np

from scipy import fftpack

X = fftpack.fft(df["D1"].to_numpy())
#freqs = fftpack.fftfreq(len(x)) * f_s

fig, ax = plt.subplots()

ax.stem(27000*len(X), np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
#ax.set_xlim(-f_s / 2, f_s / 2)
#ax.set_ylim(-5, 110)


# In[ ]:


get_ipython().system('pip install librosa')




# In[ ]:


import librosa
from scipy.io.wavfile import read
y, sr = librosa.load("fan_processed.wav")

series = np.linspace (0.0, len(y), len(y))

plt.plot(series, y)
plt.title('Wav file')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('fan_processed.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




