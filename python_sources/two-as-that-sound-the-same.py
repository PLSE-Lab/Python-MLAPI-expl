"""
The A on the a string and on the low e string sound the 
same (the fifth fret of the E string is an A). Are they really the same? Or are they separated by an
octave? Or two? Or n?

The plots suggest two things:
I. the notes are the same, so there are redundant sounds on the guitar.
II. Both strings are out of tune. After some research I found that A in the 
    4th octave should resonate at 440Hz, but these As are resonating between 500-600.
    
    That is the lowest A attainable on my guitar - if we want notes in octaves 0-3 what do we do?
    Are there other guitars that produce these low notes? Or is that what a bass guitar is for?
    
How many other fingerings on the fretboard produce the same sounds?
"""
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.fftpack import fft
import os

path_to_files = "../input/GuitarNotes/GuitarNotes"
sound_files = ["A0.wav", "Elo5.wav"]
               
for file_name in sound_files:
    spf = wave.open(os.path.join(path_to_files, file_name), 'r')
    input_signal = spf.readframes(-1)
    input_signal = np.fromstring(input_signal, 'Int16')
    freqs = fft(input_signal)
    
    f=plt.figure()
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