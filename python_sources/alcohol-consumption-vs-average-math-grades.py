# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/student-mat.csv')
df = pd.DataFrame(data)
final_g = df.loc[:, 'G3']
alc_weekend = df.loc[:, 'Walc']

#Plotting weekend alcohol consumption vs. final grades.
plt.plot([alc_weekend], [final_g])
#Setting axis labels
plt.title('Alcohol Consumption vs. Final Grades in Portugese secondary school students')
plt.xlabel('Weekend Alcohol Consumption; 1-5')
plt.ylabel('Final Grades; 0-20')
plt.show()