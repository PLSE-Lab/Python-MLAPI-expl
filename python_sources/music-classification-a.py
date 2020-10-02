# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import listdir

"""
Code from Simple Gaussian Classifier for Music Genre Classification Lab
EAAAI - 2012
Note: This code is not optimized for performance. Rather
      it was developed to be explicit so that it 
      can be ported to other programming languages.

Author: Douglas Turnbull, EAAI 2012 Model AI Assignment, Feb 2012
"""


# loads metadata (song & artist name) and audio feature vectors for all songs
# format:
#    # Count On Me - Bruno Mars
#    0.0,171.13,9.469,-28.48,57.491,-50.067,14.833,5.359,-27.228,0.973,-10.64,-7.228
#    29.775,-90.263,-47.86,14.023,13.797,189.87,50.924,-31.823,-45.63,104.501,82.114,-13.67
#
# returns data in for of "list of lists of dict" where 
#   first index is the genre and the second index corresponds to a song
#   dict contains 'song', 'artist' , and 'featureMat'

def load_data(dataDir, genres):
    data = list()
    for g in range(len(genres)):
        genreDir = dataDir + "/" + genres[g]
        data.append(list())
        sFiles = listdir(genreDir)
        for s in range(len(sFiles)):

            sFile = genreDir + "/" + sFiles[s]
            f = open(sFile)
            lines = f.readlines()
            meta = lines[0].replace("#", "").split("-")
            songDict = {'file': sFiles[s], 'song': meta[0].strip(), 'artist': meta[1].strip()}

            # read in matrix of values starting from second line in data file
            mat = list()
            for i in range(1, len(lines)):
                vec = lines[i].split(",")
                # cast to floats
                for j in range(len(vec)):
                    vec[j] = float(vec[j])
                mat.append(vec)

            songDict['featureMat'] = np.array(mat)
            data[g].append(songDict)

    return data
    
dataDir = "../input/music_data/"
genres = ['classical', 'country', 'jazz', 'pop', 'rock', 'techno']
# list of genres
data = load_data(dataDir, genres)
# get the first genre
classical = data[0]
# list of dicts, each dict is one song
for song in classical[0:3]:
    print(song['song'] + ' by ' + song['artist'] + '. Features: ' + str(song['featureMat'].shape))
    

    