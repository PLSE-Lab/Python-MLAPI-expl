import zipfile
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.fftpack import fft
import os

path_to_files = "../input/GuitarNotes/GuitarNotes"
sound_files = [f for f in os.listdir(path_to_files) if os.path.isfile(os.path.join(path_to_files, f))]

"""
zf = zipfile.ZipFile("../input/GuitarNotes.zip")
for file_name in zf.namelist():
"""

f = plt.figure()
	
for file_name in sound_files:
    if "Scale" not in file_name:
    	spf = wave.open(os.path.join(path_to_files, file_name),'r')
    	input_signal = spf.readframes(-1)
    	input_signal = np.fromstring(input_signal, 'Int16')
    	freqs = fft(input_signal)
    
    	ax = f.add_subplot(211)
    	ax.set_title('Signal Wave...')
    	ax.plot(input_signal)
    	ax = f.add_subplot(212)
    	ax.set_title("Frequency Plot")
    	ax.plot(np.abs(freqs[:1000]))
    	f.suptitle(file_name)
    	plot_name = file_name + ".png"
    	f.savefig(plot_name)
    	plt.cla()
    	plt.clf()