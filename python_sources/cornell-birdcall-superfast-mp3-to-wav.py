#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydub')
from pydub import AudioSegment
import os
from tqdm import tqdm


# ### Variables to Trim Audio (Optional)

# In[ ]:


startMin = 0
startSec = 0

endMin = 0
endSec = 10

# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000


# ### Train Audio Path and Mp3 Path

# In[ ]:


audio_path = '../input/birdsong-recognition/train_audio/'
wav_path = 'train_wav/'


# ### Save Names of Train Labels from folders

# In[ ]:


subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)


# ### Make Wav Folder to Export

# In[ ]:


if not os.path.exists(wav_path):
    os.makedirs(wav_path)


# ### Make Subfolders within Wav Folder

# In[ ]:


subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)
        if not os.path.exists(wav_path + '/' + x):
            os.makedirs(wav_path +'/'+ x)


# ### Count Items

# In[ ]:


sample_audio = []
total = 0
for x in subFolderList:
    all_files = [y for y in os.listdir(audio_path + x) if '.mp3' in y]
    total += len(all_files)
    sample_audio.append(audio_path  + x + '/'+ all_files[0])
    
    print('count: %d : %s' % (len(all_files), x ))
print(total)


# In[ ]:


def mpeg2wav(mpeg_path, targetdir=''):
    # Read Mp3
    sound = AudioSegment.from_mp3(mpeg_path)
    # Trim (10 secs)
    sound = sound[startTime:endTime]
    # Save Wav
    sound.export(f"{targetdir}.wav", format="wav")


# In[ ]:


for i, label in tqdm(enumerate(subFolderList)):
    #print(i,":",label)
    all_files = [y for y in os.listdir(audio_path + label) if '.mp3' in y]
    for file in all_files[:2]:
        mpeg2wav(audio_path + label + '/' + file, wav_path + label + "/" + file.replace(".mp3",""))


# In[ ]:




