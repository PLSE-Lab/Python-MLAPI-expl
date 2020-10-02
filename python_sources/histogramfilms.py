# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = "../input/movie_metadata.csv"
movie_data = pd.read_csv(data,sep=',')
years = movie_data['title_year']
df = pd.DataFrame({'a':years})
cnt=df.a.value_counts()

plt.plot(df.a.value_counts(),color="green")
plt.xlabel('Years')
plt.ylabel('No. of films')
plt.title('Films data per year')
plt.savefig('fig.png')

plt.bar(cnt.index,cnt.values,color="green")
plt.xlabel('Years')
plt.ylabel('No. of films')
plt.title('Films data per year')
plt.savefig('fig2.png')