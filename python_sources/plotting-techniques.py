# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input/Taekwondo_Technique_Classification_Stats.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

readings = pd.read_csv("../input/Taekwondo_Technique_Classification_Stats.csv")
readings.head()
Round = readings.iloc[2::,1:31]     # Round Kicks
plt.figure()
plt.plot(Round)
plt.title("Round Kick")

Back = readings.iloc[2::,31:56]     # Back Kicks 
plt.figure()
plt.plot(Back)
plt.title("Back Kick")

Cut = readings.iloc[2::,56:86]      # Cut Kicks 
plt.figure()
plt.plot(Cut)
plt.title("Cut Kick")

Punch = readings.iloc[2::,86:116]   # Punches
plt.figure()
plt.plot(Punch)
plt.title("Punch")