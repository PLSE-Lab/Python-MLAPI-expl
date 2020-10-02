#!/usr/bin/env python
# coding: utf-8

# # EDA OF Tess Dataset
# Only Based on 2 Speakers-a young female and older female which bias in audio features but it should be balanced the male dominant speakers on dataset SAVEE
# 
# 7 key emotions in this dataset-angry,disgust,happy,fear,neutral,Pleasant_surprise,sad

# In[ ]:


cd 


# In[ ]:


import os
import pandas as pd
TESS="input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/"
dir_list = os.listdir(TESS)
dir_list.sort()
dir_list


# In[ ]:


path = []
emotion = []

for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotion.append('female_angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion.append('female_disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion.append('female_fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion.append('female_happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion.append('female_neutral')                                
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion.append('female_surprise')               
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion.append('female_sad')
        else:
            emotion.append('Unknown')
        path.append(TESS + i + "/" + f)

TESS_df = pd.DataFrame(emotion, columns = ['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
TESS_df.labels.value_counts()


# We have around 400 files for a particular 200 from old female and 200 from young female.

# In[ ]:


pwd()


# In[ ]:


cd "input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/"


# In[ ]:


ls


# Here In this we are doing EDA on Old Talker who is speaking in angry manner and she is saying 200 words ' say the word - $word ' is what they say in all audio files

# In[ ]:


cd  OAF_angry


# In[ ]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np


# In[ ]:


data, sampling_rate = librosa.load('OAF_back_angry.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

# Lets play the audio 
#ipd.Audio('../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav')


# In[ ]:


from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# read audio samples
input_data = read('OAF_back_angry.wav')
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:1024])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot
plt.show()


# import wave
# 
# #Few observations: they are all in mono format,none in stereo format,sample rate=22050 x.shape is the length of the audio in samples(meaning more the samples,longer the audio

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_angry')


# In[ ]:


ls


# In[ ]:


cd OAF_angry/


# In[ ]:


Max=1
Min=2
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # OAF_fear

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_Fear')


# In[ ]:


cd OAF_Fear/


# In[ ]:


Max=1
Min=2
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # OAF_disgust/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_disgust')


# In[ ]:


cd OAF_disgust/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # OAF_sad/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_Sad')


# In[ ]:


cd OAF_Sad/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # OAF_Pleasant_surprise/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_Pleasant_surprise')


# In[ ]:


cd OAF_Pleasant_surprise/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # OAF_neutral/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_neutral')


# In[ ]:


cd OAF_neutral/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# #  OAF_happy/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('OAF_happy/')


# In[ ]:


cd OAF_happy/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    print(x.shape)
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # YAF_disgust/ 

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_disgust/')


# In[ ]:


cd YAF_disgust/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # YAF_pleasant_surprised/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_pleasant_surprised/')


# In[ ]:


cd YAF_pleasant_surprised/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# #   YAF_fear/     

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_fear/')


# In[ ]:


cd YAF_fear/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # YAF_sad/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_sad/')


# In[ ]:


cd YAF_sad/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# # YAF_happy/

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_happy/')


# In[ ]:


cd YAF_happy/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# In[ ]:





# # YAF_angry/ 

# In[ ]:


cd ..


# In[ ]:


File_Names=os.listdir('YAF_angry/')


# In[ ]:


cd YAF_angry/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# #    YAF_neutral/

# In[ ]:


cd ..      


# In[ ]:


File_Names=os.listdir('YAF_neutral/')


# In[ ]:


cd YAF_neutral/


# In[ ]:


Max=1
Min=3
Max_File_Name= ''
Min_File_Name= ''
sum=0
average=0
for file in File_Names:
    x, sr = librosa.load(file)
    length=librosa.get_duration(y=x, sr=sr)
    sum=sum+length
    if length > Max:
       Max=length
       Max_File_Name=file
    if length < Min:
       Min=length
       Min_File_Name=file
average=sum/200
print(average)
print(Min_File_Name,Min)
print(Max_File_Name,Max)


# Results: https://docs.google.com/spreadsheets/d/149iS0mC3PbC0Lv1OYSs_-CbGEy6ufvfkYAVDvI0Rsvk/edit#gid=0

# # FEATURE Extraction:
# - Time domain features
# These are simpler to extract and understand, like the energy of signal, zero crossing rate, maximum amplitude, minimum energy, etc.
# - Frequency based features
# are obtained by converting the time based signal into the frequency domain. Whilst they are harder to comprehend, it provides extra information that can be really handy such as pitch, rhythms, melody etc. Check this infographic below:
# ![](https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png)

# In[ ]:




